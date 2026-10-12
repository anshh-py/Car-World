(function (global) {
    var configuredApiUrl = typeof global.CARWORLD_API_BASE_URL === "string"
        ? global.CARWORLD_API_BASE_URL.replace(/\/+$/, "")
        : "";
    var apiBaseUrl = configuredApiUrl ? configuredApiUrl + "/api/v1" : "";

    function requireApiConfiguration() {
        if (!configuredApiUrl) {
            throw new Error("CarWorld API URL is not configured. Start the frontend with its environment configuration.");
        }
    }

    function safeErrorDetails(error) {
        var name = error && error.name ? String(error.name) : "Error";
        var message = error && error.message ? String(error.message) : "Request failed";
        message = message
            .replace(/(authorization|bearer|api[_-]?key|anon[_-]?key|token|password)(\s*[:=]\s*)[^\s,;]+/gi, "$1$2[redacted]")
            .slice(0, 200);
        return { name: name.slice(0, 50), message: message };
    }

    function authConfigError(url, kind, status, message, cause) {
        var diagnostic = {
            url: url,
            method: "GET",
            status: status === undefined ? null : status,
            kind: kind,
            error: safeErrorDetails(cause)
        };
        // The auth config URL contains no credentials. Never include request headers or response values.
        console.error("[CarWorld auth config] Request failed", diagnostic);
        var error = new Error(message);
        error.kind = kind;
        error.status = status;
        error.url = url;
        return error;
    }

    async function request(path, parameters, accessToken) {
        requireApiConfiguration();
        var url = new URL(apiBaseUrl + path);
        Object.keys(parameters || {}).forEach(function (key) {
            var value = parameters[key];
            if (value !== undefined && value !== null && value !== "") {
                url.searchParams.set(key, value);
            }
        });

        var headers = { Accept: "application/json" };
        if (accessToken) headers.Authorization = "Bearer " + accessToken;

        var requestUrl = url.toString();
        var response;
        try {
            response = await fetch(requestUrl, { headers: headers });
        } catch (cause) {
            if (path === "/auth/config") {
                throw authConfigError(
                    requestUrl,
                    "fetch_rejected",
                    null,
                    "The browser rejected the CarWorld auth config request before exposing a response. Inspect DevTools Network and Console for the cause.",
                    cause
                );
            }
            throw cause;
        }

        if (path === "/auth/config" && !response.ok) {
            throw authConfigError(
                response.url || requestUrl,
                "http_error",
                response.status,
                "The CarWorld backend returned HTTP " + response.status + " for its auth config.",
                new Error("HTTP " + response.status)
            );
        }

        var payload;
        try {
            payload = await response.json();
        } catch (cause) {
            if (path === "/auth/config") {
                throw authConfigError(
                    response.url || requestUrl,
                    "invalid_json",
                    response.status,
                    "The CarWorld backend returned invalid JSON for its auth config.",
                    cause
                );
            }
            payload = null;
        }
        if (!response.ok) {
            var detail = payload && payload.detail;
            if (Array.isArray(detail)) detail = detail.map(function (item) { return item.msg; }).join("; ");
            var error = new Error(detail || "The catalog request failed (" + response.status + ").");
            error.status = response.status;
            throw error;
        }

        if (path === "/auth/config" && (!payload || typeof payload !== "object" ||
            !("url" in payload) || !("anon_key" in payload) ||
            (payload.url !== null && typeof payload.url !== "string") ||
            (payload.anon_key !== null && typeof payload.anon_key !== "string"))) {
            throw authConfigError(
                response.url || requestUrl,
                "invalid_schema",
                response.status,
                "The CarWorld backend returned an unexpected auth config response.",
                new Error("Expected string-or-null url and anon_key fields")
            );
        }
        return payload;
    }

    global.CarWorldAPI = Object.freeze({
        baseUrl: apiBaseUrl,
        getCars: function (parameters) { return request("/cars", parameters); },
        searchCars: function (parameters) { return request("/cars/search", parameters); },
        getCar: function (id) { return request("/cars/" + encodeURIComponent(id)); },
        getAuthConfig: function () { return request("/auth/config"); },
        getCurrentUser: function (accessToken) { return request("/auth/me", null, accessToken); },
        chat: async function (message, history, accessToken) {
            requireApiConfiguration();
            var headers = {
                Accept: "application/json",
                "Content-Type": "application/json"
            };
            if (accessToken) headers.Authorization = "Bearer " + accessToken;
            var response = await fetch(configuredApiUrl + "/api/chat", {
                method: "POST",
                headers: headers,
                body: JSON.stringify({ message: message, history: history || [] })
            });
            var payload = await response.json().catch(function () { return null; });
            if (!response.ok) {
                var detail = payload && payload.detail;
                if (Array.isArray(detail)) detail = detail.map(function (item) { return item.msg; }).join("; ");
                var error = new Error(detail || "The chat request failed (" + response.status + ").");
                error.status = response.status;
                throw error;
            }
            return payload;
        },
        getManufacturers: function () { return request("/manufacturers"); },
        getManufacturerModels: function (name) {
            return request("/manufacturers/" + encodeURIComponent(name) + "/models");
        }
    });
})(window);
