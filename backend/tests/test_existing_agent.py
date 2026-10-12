import copy
import json
import sys
from types import SimpleNamespace

import pytest
import httpx
from groq import BadRequestError

from app.modules.chat import agent as agent_loader


class FakeCall:
	def __init__(self, name, arguments, call_id="call-1", raw_arguments=False):
		self.id = call_id
		serialized = arguments if raw_arguments else json.dumps(arguments)
		self.function = SimpleNamespace(name=name, arguments=serialized)

	def model_dump(self, exclude_none=True):
		return {
			"id": self.id,
			"type": "function",
			"function": {"name": self.function.name, "arguments": self.function.arguments},
		}


class FakeCompletions:
	def __init__(self, responses, exception=None):
		self.responses = iter(responses)
		self.exception = exception
		self.calls = []

	def create(self, **kwargs):
		self.calls.append(copy.deepcopy(kwargs))
		if self.exception is not None:
			raise self.exception
		return SimpleNamespace(choices=[SimpleNamespace(message=next(self.responses))])


def assistant(content=None, tool_calls=None):
	return SimpleNamespace(content=content, tool_calls=tool_calls or [])


@pytest.fixture
def existing_agent(monkeypatch):
	monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
	monkeypatch.setenv("TAVILY_API_KEY", "")
	monkeypatch.setattr(agent_loader.settings, "groq_api_key", "test-key-not-used")
	agent_loader._load_gradio_agent_module.cache_clear()
	module = agent_loader._load_gradio_agent_module()
	yield module
	agent_loader._load_gradio_agent_module.cache_clear()
	sys.modules.pop("carworld_gradio_agent", None)


def set_fake_client(agent, completions):
	agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_regular_question_runs_without_tools(existing_agent):
	completions = FakeCompletions([assistant("Take regular breaks and share the driving on a long trip.")])
	set_fake_client(existing_agent.agent, completions)

	answer = list(existing_agent.agent.run([{"role": "user", "content": "How can I make a long road trip more comfortable?"}]))

	assert answer == ["Take regular breaks and share the driving on a long trip."]
	assert "tools" not in completions.calls[0]
	assert completions.calls[0]["tool_choice"] == "none"
	assert "calculate_emi" not in completions.calls[0]["messages"][0]["content"]
	assert "search_car_database" not in completions.calls[0]["messages"][0]["content"]
	assert "search_car_web" not in completions.calls[0]["messages"][0]["content"]


def test_bare_scorpio_search_executes_tavily_and_returns_paired_result(existing_agent, monkeypatch):
	searches = []
	web_results = [{
		"title": "Mahindra Scorpio specifications",
		"url": "https://example.test/mahindra-scorpio",
		"content": "The report lists Scorpio variants and specifications.",
		"score": 0.92,
		"published_date": None,
	}]
	def tavily_search(**kwargs):
		searches.append(kwargs)
		return {"results": [{
			"title": web_results[0]["title"],
			"url": web_results[0]["url"],
			"content": web_results[0]["content"],
			"score": web_results[0]["score"],
		}]}
	monkeypatch.setattr(existing_agent, "tavily", SimpleNamespace(search=tavily_search))
	completions = FakeCompletions([
		assistant(tool_calls=[FakeCall("search_car_web", {"query": "Scorpio"}, "scorpio-call")]),
		assistant("The web results describe Mahindra Scorpio variants and specifications."),
	])
	set_fake_client(existing_agent.agent, completions)

	answer = list(existing_agent.agent.run([{"role": "user", "content": "Scorpio"}]))

	assert answer == ["The web results describe Mahindra Scorpio variants and specifications."]
	assert len(searches) == 1
	assert "Scorpio" in searches[0]["query"]
	assert completions.calls[0]["tool_choice"] == "auto"
	assert [tool["function"]["name"] for tool in completions.calls[0]["tools"]] == ["search_car_web"]
	tool_result = completions.calls[1]["messages"][-1]
	assert tool_result["role"] == "tool"
	assert tool_result["tool_call_id"] == "scorpio-call"
	assert tool_result["name"] == "search_car_web"
	assert json.loads(tool_result["content"])["results"] == web_results
	assert web_results[0]["content"] in tool_result["content"]


@pytest.mark.parametrize("query", [
	"Scorpio latest price",
	"Scorpio mileage",
	"Scorpio variants",
	"Scorpio reviews",
	"Scorpio 2026",
	"Compare Scorpio and XUV700",
	"Search the internet for Scorpio news",
])
def test_vehicle_information_queries_select_web_search(existing_agent, query):
	selected = existing_agent.select_tools_for_request(
		[{"role": "user", "content": query}], existing_agent.TOOLS,
	)
	assert [tool["function"]["name"] for tool in selected] == ["search_car_web"]


@pytest.mark.parametrize("query", [
	"Search our car database for Scorpio",
	"Show Scorpio cars available in our catalog",
])
def test_explicit_catalog_requests_select_database(existing_agent, query):
	selected = existing_agent.select_tools_for_request(
		[{"role": "user", "content": query}], existing_agent.TOOLS,
	)
	assert [tool["function"]["name"] for tool in selected] == ["search_car_database"]


@pytest.mark.parametrize("tavily_response", [
	{"results": []},
	RuntimeError("upstream failure"),
])
def test_web_search_failure_returns_safe_no_fabrication_message(existing_agent, monkeypatch, tavily_response):
	def tavily_search(**_kwargs):
		if isinstance(tavily_response, Exception):
			raise tavily_response
		return tavily_response
	monkeypatch.setattr(existing_agent, "tavily", SimpleNamespace(search=tavily_search))
	completions = FakeCompletions([
		assistant(tool_calls=[FakeCall("search_car_web", {"query": "Scorpio latest price"}, "failed-web-call")]),
		assistant("This must not be used because no search sources were returned."),
	])
	set_fake_client(existing_agent.agent, completions)

	answer = list(existing_agent.agent.run([{"role": "user", "content": "Scorpio latest price"}]))

	assert answer == [
		"I couldn't complete a reliable web search just now, so I can't verify those vehicle details. Please try again later."
	]
	assert len(completions.calls) == 1


def test_emi_tool_routing_and_calculation(existing_agent):
	selected = existing_agent.select_tools_for_request(
		[{"role": "user", "content": "Calculate EMI for this car loan"}],
		existing_agent.TOOLS,
	)
	assert [tool["function"]["name"] for tool in selected] == ["calculate_emi"]
	assert existing_agent.calculate_emi(100000, 0, 100)["monthly_emi"] == 1000.0

	completions = FakeCompletions([
		assistant(tool_calls=[FakeCall("calculate_emi", {
			"principal": 100000, "annual_interest_rate": 0, "tenure_months": 100,
		})]),
		assistant("Your monthly EMI is ₹1,000."),
	])
	set_fake_client(existing_agent.agent, completions)
	list(existing_agent.agent.run([{"role": "user", "content": "Calculate EMI for a 100000 loan over 100 months at 0%"}]))
	assert completions.calls[0]["tool_choice"] == "auto"
	assert [tool["function"]["name"] for tool in completions.calls[0]["tools"]] == ["calculate_emi"]
	assert completions.calls[1]["tool_choice"] == "auto"
	messages = completions.calls[1]["messages"]
	assistant_tool_message = messages[-2]
	tool_result_message = messages[-1]
	assert assistant_tool_message["role"] == "assistant"
	assert "content" not in assistant_tool_message
	assert assistant_tool_message["tool_calls"][0]["id"] == "call-1"
	assert tool_result_message["role"] == "tool"
	assert tool_result_message["tool_call_id"] == "call-1"
	assert tool_result_message["name"] == "calculate_emi"
	tool_result = tool_result_message["content"]
	assert json.loads(tool_result)["monthly_emi"] == 1000.0


def test_database_miss_uses_web_fallback(existing_agent, monkeypatch):
	queries = []
	def execute(name, arguments):
		queries.append(name)
		if name == "search_car_database":
			return {"status": "ok", "count": 0, "results": []}
		return {"status": "results_found", "results": [{"title": "Example", "content": "Car facts"}]}

	monkeypatch.setattr(existing_agent, "execute_tool", execute)
	completions = FakeCompletions([
		assistant(tool_calls=[FakeCall("search_car_database", {"query": "Honda City"})]),
		assistant("No stored match was found; web results may help."),
	])
	set_fake_client(existing_agent.agent, completions)
	list(existing_agent.agent.run([{"role": "user", "content": "Search the database for Honda City"}]))

	assert queries == ["search_car_database", "search_car_web"]
	fallback = json.loads(completions.calls[1]["messages"][-1]["content"])
	assert fallback["status"] == "web_fallback_after_database_miss"
	assert completions.calls[0]["tool_choice"] == "auto"
	assert completions.calls[1]["tool_choice"] == "auto"


def test_multiple_tool_steps_keep_calls_and_results_paired(existing_agent, monkeypatch):
	seen = []
	monkeypatch.setattr(
		existing_agent,
		"execute_tool",
		lambda name, arguments: seen.append((name, arguments)) or {"step": len(seen)},
	)
	completions = FakeCompletions([
		assistant(tool_calls=[FakeCall("calculate_emi", {"principal": 100000, "annual_interest_rate": 8, "tenure_months": 60}, "call-a")]),
		assistant(tool_calls=[FakeCall("calculate_emi", {"principal": 200000, "annual_interest_rate": 9, "tenure_months": 48}, "call-b")]),
		assistant("I completed both calculations."),
	])
	set_fake_client(existing_agent.agent, completions)

	answer = list(existing_agent.agent.run([{"role": "user", "content": "Calculate EMI for both loan options"}]))

	assert answer == ["I completed both calculations."]
	assert seen[0][0] == seen[1][0] == "calculate_emi"
	assert all(call["tool_choice"] == "auto" for call in completions.calls)
	second_step_messages = completions.calls[1]["messages"]
	assert second_step_messages[-2]["tool_calls"][0]["id"] == "call-a"
	assert second_step_messages[-1]["tool_call_id"] == "call-a"
	assert second_step_messages[-1]["name"] == "calculate_emi"
	final_messages = completions.calls[2]["messages"]
	assert final_messages[-2]["tool_calls"][0]["id"] == "call-b"
	assert final_messages[-1]["tool_call_id"] == "call-b"
	assert final_messages[-1]["name"] == "calculate_emi"


def test_tool_to_plain_turn_does_not_retain_tool_definitions_or_prompt(existing_agent):
	completions = FakeCompletions([
		assistant(tool_calls=[FakeCall("calculate_emi", {"principal": 100000, "annual_interest_rate": 0, "tenure_months": 100})]),
		assistant("The EMI is ₹1,000."),
		assistant("Regular breaks make long drives more comfortable."),
	])
	set_fake_client(existing_agent.agent, completions)
	tool_turn = [{"role": "user", "content": "Calculate EMI for a 100000 loan over 100 months at 0%"}]
	list(existing_agent.agent.run(tool_turn))
	followup = tool_turn + [{"role": "assistant", "content": "The EMI is ₹1,000."}, {"role": "user", "content": "What makes a long drive comfortable?"}]
	answer = list(existing_agent.agent.run(followup))

	assert answer == ["Regular breaks make long drives more comfortable."]
	plain_call = completions.calls[2]
	assert plain_call["tool_choice"] == "none"
	assert "tools" not in plain_call
	assert "calculate_emi" not in plain_call["messages"][0]["content"]


def test_malformed_tool_arguments_are_returned_as_paired_safe_tool_error(existing_agent, monkeypatch):
	executions = []
	monkeypatch.setattr(existing_agent, "execute_tool", lambda *_args: executions.append(True))
	completions = FakeCompletions([
		assistant(tool_calls=[FakeCall("calculate_emi", "{invalid", "bad-call", raw_arguments=True)]),
		assistant("I couldn't calculate that because the tool request was invalid."),
	])
	set_fake_client(existing_agent.agent, completions)
	list(existing_agent.agent.run([{"role": "user", "content": "Calculate EMI for a 100000 loan over 100 months at 0%"}]))

	tool_message = completions.calls[1]["messages"][-1]
	assert not executions
	assert tool_message["tool_call_id"] == "bad-call"
	assert tool_message["name"] == "calculate_emi"
	assert "error" in json.loads(tool_message["content"])


def test_unselected_registered_tool_is_not_executed(existing_agent, monkeypatch):
	executions = []
	monkeypatch.setattr(existing_agent, "execute_tool", lambda *args: executions.append(args))
	completions = FakeCompletions([assistant(tool_calls=[FakeCall("search_car_web", {"query": "latest price"})])])
	set_fake_client(existing_agent.agent, completions)

	with pytest.raises(RuntimeError, match="not enabled"):
		list(existing_agent.agent.run([{"role": "user", "content": "Calculate EMI for a 100000 loan over 100 months at 0%"}]))
	assert not executions


def test_unrelated_groq_400_is_not_retried(existing_agent):
	request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
	error = BadRequestError(
		"Invalid request",
		response=httpx.Response(400, request=request, json={"error": {"code": "invalid_parameter"}}),
		body={"error": {"code": "invalid_parameter"}},
	)
	completions = FakeCompletions([], exception=error)
	set_fake_client(existing_agent.agent, completions)

	with pytest.raises(BadRequestError):
		list(existing_agent.agent.run([{"role": "user", "content": "What is the Honda City?"}]))
	assert len(completions.calls) == 1


def test_postgres_search_uses_parameterized_query(existing_agent, monkeypatch):
	from car_database import search_car_database

	class Cursor:
		def __enter__(self): return self
		def __exit__(self, *_args): return False
		def execute(self, query, params):
			assert "ILIKE %s" in query
			assert " AND " in query
			assert params == ("%Honda%",) * 3 + ("%City%",) * 3
		def fetchall(self):
			return [("Honda", "City", "V", 1200000, "Petrol", "Manual", "Sedan")]

	class Connection:
		def __enter__(self): return self
		def __exit__(self, *_args): return False
		def cursor(self): return Cursor()

	monkeypatch.setattr("car_database.psycopg.connect", lambda **_kwargs: Connection())
	result = search_car_database("Honda City")
	assert result["status"] == "ok"
	assert result["count"] == 1
	assert result["results"][0]["make"] == "Honda"


def test_postgres_search_matches_make_and_model_across_columns(existing_agent, monkeypatch):
	from car_database import search_car_database

	class Cursor:
		def __enter__(self): return self
		def __exit__(self, *_args): return False
		def execute(self, query, params):
			assert query.count("ILIKE %s") == 6
			assert " AND " in query
			assert params == ("%Mahindra%",) * 3 + ("%Scorpio%",) * 3
		def fetchall(self):
			return [("Mahindra", "Scorpio", "S11", 1540081, "Diesel", "Manual", "SUV")]

	class Connection:
		def __enter__(self): return self
		def __exit__(self, *_args): return False
		def cursor(self): return Cursor()

	monkeypatch.setattr("car_database.psycopg.connect", lambda **_kwargs: Connection())
	result = search_car_database("Search database for Mahindra Scorpio")
	assert result["status"] == "ok"
	assert result["count"] == 1
	assert result["results"][0]["model"] == "Scorpio"
