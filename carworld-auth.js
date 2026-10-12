(function (global) {
    var client = null;
    var currentSession = null;
    var elements = null;
    var initializationPromise = null;
    var initializationError = null;

    function safeAuthErrorDetails(error) {
        var name = error && error.name ? String(error.name) : "Error";
        var message = error && error.message ? String(error.message) : "Initialization failed";
        message = message
            .replace(/(authorization|bearer|api[_-]?key|anon[_-]?key|token|password)(\s*[:=]\s*)[^\s,;]+/gi, "$1$2[redacted]")
            .slice(0, 200);
        return { name: name.slice(0, 50), message: message };
    }

    function setMessage(message, kind) {
        elements.message.className = "alert alert-" + kind + " mt-3 mb-2";
        elements.message.textContent = message;
        elements.message.hidden = false;
    }

    function updateAuthUI(user, state) {
        var signedIn = Boolean(user);
        elements.googleLink.hidden = signedIn;
        elements.logoutButton.hidden = !signedIn;
        elements.userText.hidden = !signedIn;

        if (signedIn) {
            var identity = state && state.verifiedUser;
            elements.userText.textContent = "Signed in as " +
                ((identity && identity.email) || user.email || user.id);
            if (state && state.error) {
                setMessage(state.error, "warning");
            } else if (state && state.loading) {
                setMessage("Checking your CarWorld session...", "info");
            } else if (identity) {
                setMessage("Your CarWorld session is verified.", "success");
            } else {
                setMessage("Supabase session is active.", "info");
            }
            return;
        }

        elements.userText.textContent = "";
        setMessage((state && state.error) || "You are signed out.", state && state.error ? "danger" : "secondary");
    }

    function createAuthUI() {
        var modal = document.getElementById("loginmodal");
        if (!modal) return null;

        var body = modal.querySelector(".modal-body");
        var googleLink = body.querySelector("[data-carworld-google-login]") ||
            body.querySelector('a[href*="accounts.google.com"]');
        if (!googleLink) return null;

        body.querySelectorAll("form").forEach(function (form) { form.hidden = true; });
        var signupLink = document.getElementById("signup");
        if (signupLink) {
            var signupContainer = signupLink.closest(".text-center");
            if (signupContainer) signupContainer.hidden = true;
        }
        var signupModal = document.getElementById("loginmodal1");
        if (signupModal) signupModal.hidden = true;

        var message = document.createElement("div");
        message.id = "carworld-auth-message";
        message.className = "alert alert-secondary mt-3 mb-2";
        message.setAttribute("role", "status");
        message.textContent = "Loading authentication...";

        var userText = document.createElement("p");
        userText.id = "carworld-auth-user";
        userText.className = "text-center mt-3";
        userText.hidden = true;

        var logoutButton = document.createElement("button");
        logoutButton.id = "carworld-logout";
        logoutButton.className = "btn btn-outline-secondary btn-block mt-2";
        logoutButton.type = "button";
        logoutButton.textContent = "Sign out";
        logoutButton.hidden = true;

        var title = body.querySelector(".modal-title");
        if (title) title.insertAdjacentElement("afterend", message);
        else body.prepend(message);
        message.insertAdjacentElement("afterend", userText);
        googleLink.insertAdjacentElement("afterend", logoutButton);

        googleLink.href = "#";
        googleLink.addEventListener("click", function (event) {
            event.preventDefault();
            signInWithGoogle();
        });
        logoutButton.addEventListener("click", signOut);

        return { message: message, userText: userText, googleLink: googleLink, logoutButton: logoutButton };
    }

    function loadSupabaseSdk() {
        return new Promise(function (resolve, reject) {
            if (global.supabase && typeof global.supabase.createClient === "function") {
                resolve();
                return;
            }

            var script = document.createElement("script");
            script.src = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2";
            script.onload = function () {
                if (global.supabase && typeof global.supabase.createClient === "function") resolve();
                else reject(new Error("Supabase JavaScript client did not initialize."));
            };
            script.onerror = function () { reject(new Error("Could not load the Supabase JavaScript client.")); };
            document.head.appendChild(script);
        });
    }

    async function verifyBackendSession(session) {
        if (!session || !global.CarWorldAPI) return;
        try {
            var verifiedUser = await global.CarWorldAPI.getCurrentUser(session.access_token);
            if (currentSession !== session) return;
            updateAuthUI(session.user, { verifiedUser: verifiedUser });
        } catch (error) {
            if (currentSession !== session) return;
            updateAuthUI(session.user, {
                error: "Your Supabase session is active, but CarWorld could not verify it: " + error.message
            });
        }
    }

    function handleAuthState(event, session) {
        currentSession = session || null;
        if (!currentSession) {
            updateAuthUI(null, {});
            return;
        }

        updateAuthUI(currentSession.user, { loading: true });
        global.setTimeout(function () { verifyBackendSession(currentSession); }, 0);
    }

    async function signInWithGoogle() {
        if (!client) {
            await initialize();
            if (!client) {
                setMessage(initializationError || "Google sign-in is not ready. Please try again. Guest chat remains available.", "warning");
                return;
            }
        }

        try {
            setMessage("Redirecting to Google through Supabase...", "info");
            var result = await client.auth.signInWithOAuth({
                provider: "google",
                options: { redirectTo: new URL("./index.html", global.location.href).toString() }
            });
            if (result.error) setMessage(result.error.message, "danger");
        } catch (error) {
            setMessage("Google sign-in failed: " + error.message, "danger");
        }
    }

    async function signOut() {
        if (!client) return;
        elements.logoutButton.disabled = true;
        try {
            var result = await client.auth.signOut({ scope: "local" });
            if (result.error) setMessage(result.error.message, "danger");
        } catch (error) {
            setMessage("Sign out failed: " + error.message, "danger");
        } finally {
            elements.logoutButton.disabled = false;
        }
    }

    async function getCurrentUser() {
        if (!client) return null;
        var result = await client.auth.getUser();
        if (result.error) throw result.error;
        return result.data.user;
    }

    async function getSession() {
        if (!client) return null;
        var result = await client.auth.getSession();
        if (result.error) throw result.error;
        return result.data.session;
    }

    async function initialize() {
        if (client || initializationPromise) return initializationPromise;

        initializationPromise = (async function () {
            elements = elements || createAuthUI();
            if (!elements) return;
            initializationError = null;

            var config;
            try {
                config = await global.CarWorldAPI.getAuthConfig();
            } catch (error) {
                if (error && error.kind === "http_error") {
                    initializationError = "The CarWorld backend returned an HTTP error while loading sign-in settings. Please try again later. Guest chat remains available.";
                } else if (error && (error.kind === "invalid_json" || error.kind === "invalid_schema")) {
                    initializationError = "The CarWorld backend returned an unreadable sign-in configuration. Please try again later. Guest chat remains available.";
                } else if (error && error.kind === "fetch_rejected") {
                    initializationError = "The browser rejected the CarWorld sign-in settings request before returning a response. Check the browser Network and Console panels for the cause. Guest chat remains available.";
                } else {
                    initializationError = "CarWorld couldn't load sign-in settings. Check the browser Network and Console panels, then try again. Guest chat remains available.";
                }
                updateAuthUI(null, { error: initializationError });
                return;
            }

            if (!config || !config.url || !config.anon_key) {
                initializationError = "Google sign-in is unavailable because SUPABASE_URL and SUPABASE_ANON_KEY are missing from the project-root .env file. Guest chat remains available.";
                updateAuthUI(null, { error: initializationError });
                return;
            }

            try {
                await loadSupabaseSdk();
                client = global.supabase.createClient(config.url, config.anon_key, {
                    auth: {
                        persistSession: true,
                        autoRefreshToken: true,
                        detectSessionInUrl: true
                    }
                });

                client.auth.onAuthStateChange(handleAuthState);
                var sessionResult = await client.auth.getSession();
                if (sessionResult.error) throw sessionResult.error;
                handleAuthState("INITIAL_SESSION", sessionResult.data.session);
            } catch (error) {
                client = null;
                console.error("[CarWorld auth] Supabase SDK or client initialization failed", {
                    url: "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2",
                    kind: "supabase_sdk_initialization",
                    error: safeAuthErrorDetails(error)
                });
                initializationError = "CarWorld loaded its sign-in settings but couldn't initialize Google sign-in. Check your connection and try again. Guest chat remains available.";
                updateAuthUI(null, { error: initializationError });
            }
        })();

        try {
            return await initializationPromise;
        } finally {
            initializationPromise = null;
        }
    }

    global.CarWorldAuth = Object.freeze({
        signInWithGoogle: signInWithGoogle,
        signOut: signOut,
        getCurrentUser: getCurrentUser,
        getSession: getSession,
        handleAuthState: handleAuthState,
        updateAuthUI: updateAuthUI
    });

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, { once: true });
    } else {
        initialize();
    }
})(window);
