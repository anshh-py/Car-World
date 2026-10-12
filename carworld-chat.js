(function (global) {
    var messages = [];
    var nodes = {};
    var isOpen = false;
    var isSending = false;

    function query(selector) { return document.querySelector(selector); }

    function setStatus(message, isError) {
        nodes.status.textContent = message || "";
        nodes.status.classList.toggle("is-error", Boolean(isError));
    }

    function scrollToLatest() {
        nodes.messages.scrollTop = nodes.messages.scrollHeight;
    }

    function appendMessage(role, text) {
        var row = document.createElement("article");
        row.className = "cw-chat-message cw-chat-message-" + role;

        var label = document.createElement("span");
        label.className = "cw-chat-message-label";
        label.textContent = role === "user" ? "You" : "CarWorld AI";

        var bubble = document.createElement("div");
        bubble.className = "cw-chat-message-bubble";
        bubble.textContent = text;
        row.appendChild(label);
        row.appendChild(bubble);
        nodes.messages.appendChild(row);
        scrollToLatest();
    }

    function restoreWelcome() {
        nodes.messages.replaceChildren(nodes.welcome);
        nodes.welcome.hidden = false;
    }

    function setOpen(open, focus) {
        isOpen = Boolean(open);
        nodes.panel.hidden = !isOpen;
        nodes.launcher.setAttribute("aria-expanded", String(isOpen));
        nodes.launcher.setAttribute("aria-label", isOpen ? "Minimize CarWorld AI chat" : "Open CarWorld AI chat");

        if (isOpen && focus) {
            global.requestAnimationFrame(function () { nodes.input.focus(); });
        } else if (!isOpen && focus) {
            nodes.launcher.focus();
        }
    }

    function setComposerState() {
        var hasText = nodes.input.value.trim().length > 0;
        nodes.send.disabled = isSending || !hasText;
        nodes.input.disabled = isSending;
        nodes.input.style.height = "auto";
        nodes.input.style.height = Math.min(nodes.input.scrollHeight, 88) + "px";
    }

    async function optionalAccessToken() {
        if (!global.CarWorldAuth || typeof global.CarWorldAuth.getSession !== "function") return null;
        try {
            var session = await global.CarWorldAuth.getSession();
            return session && session.access_token ? session.access_token : null;
        } catch (error) {
            // Authentication is optional for chat. A session refresh issue should
            // not prevent a visitor from continuing as a guest.
            return null;
        }
    }

    async function requestReply(message, history, accessToken) {
        try {
            return await global.CarWorldAPI.chat(message, history, accessToken);
        } catch (error) {
            if (accessToken && error.status === 401) {
                setStatus("Your sign-in expired. Continuing as a guest...");
                return global.CarWorldAPI.chat(message, history, null);
            }
            throw error;
        }
    }

    async function send(event) {
        event.preventDefault();
        if (isSending) return;

        var message = nodes.input.value.trim();
        if (!message) {
            setStatus("Type a question to get started.", true);
            nodes.input.focus();
            return;
        }

        nodes.welcome.hidden = true;
        var history = messages.slice(-20);
        messages.push({ role: "user", content: message });
        appendMessage("user", message);
        nodes.input.value = "";
        isSending = true;
        nodes.typing.hidden = false;
        setStatus("");
        setComposerState();

        try {
            var accessToken = await optionalAccessToken();
            var result = await requestReply(message, history, accessToken);
            messages.push({ role: "assistant", content: result.reply });
            appendMessage("assistant", result.reply);
            setStatus("");
        } catch (error) {
            // Keep the failed turn visible, but exclude it from later model history.
            messages.pop();
            setStatus(error.message || "CarWorld AI could not reply. Please try again.", true);
        } finally {
            isSending = false;
            nodes.typing.hidden = true;
            setComposerState();
            if (isOpen) nodes.input.focus();
        }
    }

    function clearConversation() {
        if (isSending) return;
        if (messages.length && !global.confirm("Clear this CarWorld AI conversation?")) return;
        messages = [];
        restoreWelcome();
        setStatus("");
        nodes.input.value = "";
        setComposerState();
        nodes.input.focus();
    }

    function usePrompt(event) {
        var prompt = event.currentTarget.getAttribute("data-chat-prompt");
        if (!prompt || isSending) return;
        nodes.input.value = prompt;
        setComposerState();
        nodes.form.requestSubmit();
    }

    function onKeyDown(event) {
        if (event.key === "Escape" && isOpen) {
            event.preventDefault();
            setOpen(false, true);
        }
        if (event.target === nodes.input && event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (!nodes.send.disabled) nodes.form.requestSubmit();
        }
    }

    function initialize() {
        var widget = query("[data-chat-widget]");
        if (!widget || !global.CarWorldAPI) return;

        nodes = {
            launcher: widget.querySelector("[data-chat-open]"),
            panel: widget.querySelector("[data-chat-panel]"),
            close: widget.querySelector("[data-chat-close]"),
            clear: widget.querySelector("[data-chat-clear]"),
            messages: widget.querySelector("[data-chat-messages]"),
            welcome: widget.querySelector("[data-chat-welcome]"),
            typing: widget.querySelector("[data-chat-typing]"),
            status: widget.querySelector("[data-chat-status]"),
            form: widget.querySelector("[data-chat-form]"),
            input: widget.querySelector("[data-chat-form] textarea"),
            send: widget.querySelector("[data-chat-send]")
        };

        nodes.launcher.addEventListener("click", function () { setOpen(!isOpen, true); });
        nodes.close.addEventListener("click", function () { setOpen(false, true); });
        nodes.clear.addEventListener("click", clearConversation);
        nodes.form.addEventListener("submit", send);
        nodes.input.addEventListener("input", setComposerState);
        nodes.input.addEventListener("keydown", onKeyDown);
        document.addEventListener("keydown", onKeyDown);
        widget.querySelectorAll("[data-chat-prompt]").forEach(function (button) {
            button.addEventListener("click", usePrompt);
        });
        setComposerState();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, { once: true });
    } else {
        initialize();
    }
})(window);
