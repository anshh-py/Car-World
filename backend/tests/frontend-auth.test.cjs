const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const projectRoot = path.resolve(__dirname, "../..");
const authSource = fs.readFileSync(path.join(projectRoot, "carworld-auth.js"), "utf8");
const apiSource = fs.readFileSync(path.join(projectRoot, "carworld-api.js"), "utf8");

function createElement() {
  return {
    hidden: false,
    textContent: "",
    className: "",
    setAttribute() {},
    addEventListener() {},
    insertAdjacentElement() {},
    closest() { return null; },
  };
}

function loadAuth(getAuthConfig, withSupabaseSdk = true) {
  const createdElements = [];
  const logs = [];
  const googleLink = createElement();
  const body = {
    querySelector(selector) {
      return selector === ".modal-title" ? null : googleLink;
    },
    querySelectorAll() { return []; },
    prepend() {},
  };
  const modal = { querySelector() { return body; } };
  const document = {
    readyState: "complete",
    getElementById(id) {
      if (id === "loginmodal") return modal;
      return null;
    },
    createElement() {
      const element = createElement();
      createdElements.push(element);
      return element;
    },
    head: {
      appendChild(script) {
        if (script.onerror) setImmediate(script.onerror);
      },
    },
    addEventListener() {},
  };
  const calls = { config: 0, oauth: 0 };
  const window = {
    document,
    location: { href: "http://127.0.0.1:5500/index.html" },
    setTimeout,
    CarWorldAPI: {
      async getAuthConfig() {
        calls.config += 1;
        return getAuthConfig(calls.config);
      },
    },
    supabase: withSupabaseSdk ? {
      createClient() {
        return {
          auth: {
            onAuthStateChange() {},
            async getSession() { return { data: { session: null }, error: null }; },
            async signInWithOAuth() { calls.oauth += 1; return { error: null }; },
          },
        };
      },
    } : undefined,
  };
  vm.runInNewContext(authSource, {
    window,
    document,
    URL,
    console: { error(...args) { logs.push(args); } },
    setTimeout,
  });
  return {
    auth: window.CarWorldAuth,
    calls,
    logs,
    get message() { return createdElements.find((element) => element.id === "carworld-auth-message"); },
  };
}

test("backend config fetch errors are not reported as missing Supabase credentials and can be retried", async () => {
  const state = loadAuth((attempt) => {
    if (attempt === 1) {
      const error = new TypeError("Failed to fetch");
      error.kind = "fetch_rejected";
      throw error;
    }
    return { url: "https://supabase.example.invalid", anon_key: "test-only" };
  });
  await new Promise(setImmediate);

  assert.match(state.message.textContent, /rejected.*before returning a response/i);
  assert.doesNotMatch(state.message.textContent, /SUPABASE_URL.*missing/i);
  await state.auth.signInWithGoogle();
  assert.equal(state.calls.config, 2);
  assert.equal(state.calls.oauth, 1);
});

test("only an empty backend auth config reports missing Supabase settings", async () => {
  const state = loadAuth(() => ({ url: null, anon_key: null }));
  await new Promise(setImmediate);

  assert.match(state.message.textContent, /SUPABASE_URL and SUPABASE_ANON_KEY are missing/i);
  assert.match(state.message.textContent, /Guest chat remains available/i);
});

test("Supabase SDK loading failure is diagnosed separately from backend configuration", async () => {
  const state = loadAuth(() => ({ url: "https://supabase.example.invalid", anon_key: "test-only" }), false);
  await new Promise((resolve) => setTimeout(resolve, 25));

  assert.match(state.message.textContent, /loaded its sign-in settings but couldn't initialize Google sign-in/i);
  assert.equal(state.logs[0][1].kind, "supabase_sdk_initialization");
  assert.equal(state.logs[0][1].url, "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2");
});

function loadApi(fetchImpl, apiOrigin) {
  const logs = [];
  if (arguments.length < 2) apiOrigin = "http://127.0.0.1:8000";
  const window = {};
  if (apiOrigin !== undefined) window.CARWORLD_API_BASE_URL = apiOrigin;
  vm.runInNewContext(apiSource, {
    window,
    URL,
    fetch: fetchImpl,
    console: { error(...args) { logs.push(args); } },
  });
  return { api: window.CarWorldAPI, logs };
}

test("missing frontend API configuration fails with an actionable error before fetch", async () => {
  let fetchCalled = false;
  const state = loadApi(async () => { fetchCalled = true; }, undefined);
  await assert.rejects(state.api.getCar(261), /CarWorld API URL is not configured.*environment configuration/i);
  assert.equal(fetchCalled, false);
});

test("valid frontend API configuration builds a backend URL for a car detail request", async () => {
  let requestUrl;
  const state = loadApi(async (url) => {
    requestUrl = url;
    return { ok: true, status: 200, json: async () => ({ id: 261, manufacturer: "Aston Martin", model: "Db 11" }) };
  });
  const car = await state.api.getCar(261);
  assert.equal(requestUrl, "http://127.0.0.1:8000/api/v1/cars/261");
  assert.equal(car.id, 261);
  assert.equal(car.manufacturer, "Aston Martin");
});

test("auth config API reports fetch rejection with sanitized URL, method, and no headers", async () => {
  const state = loadApi(async () => { throw new TypeError("Failed to fetch"); });
  await assert.rejects(state.api.getAuthConfig(), (error) => error.kind === "fetch_rejected");
  const [label, details] = state.logs[0];
  assert.match(label, /CarWorld auth config/);
  assert.equal(details.url, "http://127.0.0.1:8000/api/v1/auth/config");
  assert.equal(details.method, "GET");
  assert.equal(details.status, null);
  assert.equal("headers" in details, false);
});

test("auth config API distinguishes HTTP errors from invalid JSON and schema", async (t) => {
  await t.test("HTTP status", async () => {
    const state = loadApi(async () => ({ ok: false, status: 503 }));
    await assert.rejects(state.api.getAuthConfig(), (error) => error.kind === "http_error" && error.status === 503);
    assert.equal(state.logs[0][1].status, 503);
  });

  await t.test("invalid JSON", async () => {
    const state = loadApi(async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError("bad JSON"); } }));
    await assert.rejects(state.api.getAuthConfig(), (error) => error.kind === "invalid_json" && error.status === 200);
  });

  await t.test("unexpected schema", async () => {
    const state = loadApi(async () => ({ ok: true, status: 200, json: async () => ({ url: "https://example.invalid" }) }));
    await assert.rejects(state.api.getAuthConfig(), (error) => error.kind === "invalid_schema" && error.status === 200);
  });
});
