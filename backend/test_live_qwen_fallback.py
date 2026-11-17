
import getpass
import httpx

from groq import RateLimitError
from app.modules.chat.providers import (
    Provider,
    ProviderResponse,
    ProviderRouter,
    Qwen3Provider,
)


class SimulatedGroq(Provider):
    """Simulates Groq 429 once, then simulates recovery."""

    def __init__(self):
        self.calls = 0

    @property
    def name(self):
        return "groq"

    def supports_tools(self):
        return True

    def chat_completion(
        self, messages, tools=None, tool_choice="auto", temperature=0.2
    ):
        self.calls += 1

        if self.calls == 1:
            response = httpx.Response(
                429,
                request=httpx.Request(
                    "POST", "https://api.groq.com"
                ),
            )
            raise RateLimitError(
                "Simulated Groq rate limit",
                response=response,
                body={},
            )

        return ProviderResponse(
            content="Simulated Groq recovery successful.",
            provider_name="groq",
        )


def main():
    print("\n=== CAR WORLD LIVE QWEN3 FALLBACK TEST ===\n")

    space_id = input(
        "Hugging Face Space ID [ansh9057/carworld-qwen3-inference]: "
    ).strip() or "ansh9057/carworld-qwen3-inference"

    token = getpass.getpass(
        "Hugging Face token (hidden input): "
    ).strip()

    if not token:
        raise SystemExit("FAIL: Hugging Face token is required.")

    print("\nInitializing real Qwen3 provider...")

    qwen = Qwen3Provider(
        space_id=space_id,
        hf_token=token,
        timeout=120,
        supports_tool_calling=False,
    )

    groq = SimulatedGroq()

    # Zero-second cooldown makes recovery test immediate.
    router = ProviderRouter(
        groq_provider=groq,
        qwen3_provider=qwen,
        cooldown_seconds=0,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Explain the difference between petrol and diesel "
                "engines in three concise points."
            ),
        }
    ]

    print("\nTEST 1: Simulating Groq HTTP 429...")
    fallback = router.chat_completion(
        messages=messages,
        tools=None,
        tool_choice="none",
        temperature=0.2,
    )

    if fallback.provider_name != "qwen3_fallback":
        raise SystemExit(
            f"FAIL: Expected Qwen3, got {fallback.provider_name}"
        )

    if not fallback.content.strip():
        raise SystemExit("FAIL: Qwen3 returned an empty response.")

    print("PASS: Actual fine-tuned Qwen3 fallback worked.")
    print("\nQwen3 response:\n")
    print(fallback.content)

    print("\nTEST 2: Checking Groq recovery after cooldown...")

    recovered = router.chat_completion(
        messages=messages,
        tools=None,
        tool_choice="none",
        temperature=0.2,
    )

    if recovered.provider_name != "groq":
        raise SystemExit(
            f"FAIL: Expected Groq recovery, got {recovered.provider_name}"
        )

    print("PASS: Groq primary provider recovered.")
    print(recovered.content)

    print("\n=== ALL TESTS PASSED ===")
    print("Real Qwen3 inference: PASS")
    print("Groq 429 fallback routing: PASS")
    print("Groq recovery routing: PASS")


if __name__ == "__main__":
    main()
