import { jsx, jsxs } from "react/jsx-runtime";
import { PassThrough } from "node:stream";
import { createReadableStreamFromReadable } from "@remix-run/node";
import { RemixServer, Outlet, Meta, Links, ScrollRestoration, Scripts } from "@remix-run/react";
import * as isbotModule from "isbot";
import { renderToPipeableStream } from "react-dom/server";
import { useState, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
const ABORT_DELAY = 5e3;
function handleRequest(request, responseStatusCode, responseHeaders, remixContext, loadContext) {
  let prohibitOutOfOrderStreaming = isBotRequest(request.headers.get("user-agent")) || remixContext.isSpaMode;
  return prohibitOutOfOrderStreaming ? handleBotRequest(
    request,
    responseStatusCode,
    responseHeaders,
    remixContext
  ) : handleBrowserRequest(
    request,
    responseStatusCode,
    responseHeaders,
    remixContext
  );
}
function isBotRequest(userAgent) {
  if (!userAgent) {
    return false;
  }
  if ("isbot" in isbotModule && typeof isbotModule.isbot === "function") {
    return isbotModule.isbot(userAgent);
  }
  if ("default" in isbotModule && typeof isbotModule.default === "function") {
    return isbotModule.default(userAgent);
  }
  return false;
}
function handleBotRequest(request, responseStatusCode, responseHeaders, remixContext) {
  return new Promise((resolve, reject) => {
    let shellRendered = false;
    const { pipe, abort } = renderToPipeableStream(
      /* @__PURE__ */ jsx(
        RemixServer,
        {
          context: remixContext,
          url: request.url,
          abortDelay: ABORT_DELAY
        }
      ),
      {
        onAllReady() {
          shellRendered = true;
          const body = new PassThrough();
          const stream = createReadableStreamFromReadable(body);
          responseHeaders.set("Content-Type", "text/html");
          resolve(
            new Response(stream, {
              headers: responseHeaders,
              status: responseStatusCode
            })
          );
          pipe(body);
        },
        onShellError(error) {
          reject(error);
        },
        onError(error) {
          responseStatusCode = 500;
          if (shellRendered) {
            console.error(error);
          }
        }
      }
    );
    setTimeout(abort, ABORT_DELAY);
  });
}
function handleBrowserRequest(request, responseStatusCode, responseHeaders, remixContext) {
  return new Promise((resolve, reject) => {
    let shellRendered = false;
    const { pipe, abort } = renderToPipeableStream(
      /* @__PURE__ */ jsx(
        RemixServer,
        {
          context: remixContext,
          url: request.url,
          abortDelay: ABORT_DELAY
        }
      ),
      {
        onShellReady() {
          shellRendered = true;
          const body = new PassThrough();
          const stream = createReadableStreamFromReadable(body);
          responseHeaders.set("Content-Type", "text/html");
          resolve(
            new Response(stream, {
              headers: responseHeaders,
              status: responseStatusCode
            })
          );
          pipe(body);
        },
        onShellError(error) {
          reject(error);
        },
        onError(error) {
          responseStatusCode = 500;
          if (shellRendered) {
            console.error(error);
          }
        }
      }
    );
    setTimeout(abort, ABORT_DELAY);
  });
}
const entryServer = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: handleRequest
}, Symbol.toStringTag, { value: "Module" }));
const links = () => [
  { rel: "stylesheet", href: "/tailwind.css" }
];
function Layout({ children }) {
  return /* @__PURE__ */ jsxs("html", { lang: "en", className: "h-full", children: [
    /* @__PURE__ */ jsxs("head", { children: [
      /* @__PURE__ */ jsx("meta", { charSet: "utf-8" }),
      /* @__PURE__ */ jsx("meta", { name: "viewport", content: "width=device-width, initial-scale=1" }),
      /* @__PURE__ */ jsx(Meta, {}),
      /* @__PURE__ */ jsx(Links, {})
    ] }),
    /* @__PURE__ */ jsxs("body", { className: "h-full bg-gray-950 text-gray-100", children: [
      children,
      /* @__PURE__ */ jsx(ScrollRestoration, {}),
      /* @__PURE__ */ jsx(Scripts, {})
    ] })
  ] });
}
function App() {
  return /* @__PURE__ */ jsx(Outlet, {});
}
const route0 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Layout,
  default: App,
  links
}, Symbol.toStringTag, { value: "Module" }));
const API_URL = "http://localhost:8000";
function Index() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState("");
  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const abortRef = useRef(null);
  const bottomRef = useRef(null);
  const handleSubmit = useCallback(
    async (e) => {
      var _a, _b;
      e.preventDefault();
      if (!query.trim() || status === "streaming") return;
      (_a = abortRef.current) == null ? void 0 : _a.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setResult("");
      setErrorMsg("");
      setStatus("streaming");
      try {
        const response = await fetch(`${API_URL}/research`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query.trim() }),
          signal: controller.signal
        });
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") {
              setStatus("done");
              return;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.error) {
                setErrorMsg(parsed.error);
                setStatus("error");
                return;
              }
              if (parsed.text) {
                setResult((prev) => prev + parsed.text);
                (_b = bottomRef.current) == null ? void 0 : _b.scrollIntoView({ behavior: "smooth" });
              }
            } catch {
            }
          }
        }
        setStatus("done");
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") return;
        setErrorMsg(err instanceof Error ? err.message : "Unknown error");
        setStatus("error");
      }
    },
    [query, status]
  );
  const handleStop = () => {
    var _a;
    (_a = abortRef.current) == null ? void 0 : _a.abort();
    setStatus("done");
  };
  return /* @__PURE__ */ jsxs("div", { className: "min-h-full flex flex-col", children: [
    /* @__PURE__ */ jsx("header", { className: "border-b border-gray-800 px-6 py-4", children: /* @__PURE__ */ jsxs("div", { className: "max-w-4xl mx-auto flex items-center gap-3", children: [
      /* @__PURE__ */ jsx("div", { className: "text-2xl", children: "✝" }),
      /* @__PURE__ */ jsxs("div", { children: [
        /* @__PURE__ */ jsx("h1", { className: "text-xl font-semibold text-white", children: "Tabor Research" }),
        /* @__PURE__ */ jsx("p", { className: "text-xs text-gray-500", children: "Catholic SRI · Stocks · ETFs" })
      ] })
    ] }) }),
    /* @__PURE__ */ jsxs("main", { className: "flex-1 flex flex-col max-w-4xl mx-auto w-full px-6 py-8 gap-6", children: [
      /* @__PURE__ */ jsxs("form", { onSubmit: handleSubmit, className: "flex flex-col gap-3", children: [
        /* @__PURE__ */ jsx(
          "label",
          {
            htmlFor: "query",
            className: "text-sm font-medium text-gray-400 uppercase tracking-wide",
            children: "Research Query"
          }
        ),
        /* @__PURE__ */ jsxs("div", { className: "flex gap-3", children: [
          /* @__PURE__ */ jsx(
            "input",
            {
              id: "query",
              type: "text",
              value: query,
              onChange: (e) => setQuery(e.target.value),
              placeholder: 'e.g. "Analyze SPY ETF for Catholic SRI compliance"',
              disabled: status === "streaming",
              className: "flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50 transition"
            }
          ),
          status === "streaming" ? /* @__PURE__ */ jsx(
            "button",
            {
              type: "button",
              onClick: handleStop,
              className: "px-5 py-3 bg-red-700 hover:bg-red-600 text-white rounded-lg font-medium transition",
              children: "Stop"
            }
          ) : /* @__PURE__ */ jsx(
            "button",
            {
              type: "submit",
              disabled: !query.trim(),
              className: "px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg font-medium transition",
              children: "Research"
            }
          )
        ] })
      ] }),
      status === "streaming" && /* @__PURE__ */ jsxs("div", { className: "flex items-center gap-2 text-sm text-blue-400", children: [
        /* @__PURE__ */ jsx("span", { className: "inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse" }),
        "Researching…"
      ] }),
      status === "error" && errorMsg && /* @__PURE__ */ jsx("div", { className: "rounded-lg bg-red-950 border border-red-800 px-4 py-3 text-red-300 text-sm", children: errorMsg }),
      result && /* @__PURE__ */ jsx("article", { className: "flex-1 bg-gray-900 border border-gray-800 rounded-xl p-6 prose prose-invert prose-sm max-w-none", children: /* @__PURE__ */ jsx(ReactMarkdown, { children: result }) }),
      /* @__PURE__ */ jsx("div", { ref: bottomRef })
    ] })
  ] });
}
const route1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: Index
}, Symbol.toStringTag, { value: "Module" }));
const serverManifest = { "entry": { "module": "/assets/entry.client-CqW2bppk.js", "imports": ["/assets/index-BJHAE5s4.js", "/assets/components-XrQPQ4fj.js"], "css": [] }, "routes": { "root": { "id": "root", "parentId": void 0, "path": "", "index": void 0, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasErrorBoundary": false, "module": "/assets/root-DUP_5CFO.js", "imports": ["/assets/index-BJHAE5s4.js", "/assets/components-XrQPQ4fj.js"], "css": [] }, "routes/_index": { "id": "routes/_index", "parentId": "root", "path": void 0, "index": true, "caseSensitive": void 0, "hasAction": false, "hasLoader": false, "hasClientAction": false, "hasClientLoader": false, "hasErrorBoundary": false, "module": "/assets/_index-C0tRh34m.js", "imports": ["/assets/index-BJHAE5s4.js"], "css": [] } }, "url": "/assets/manifest-29383179.js", "version": "29383179" };
const mode = "production";
const assetsBuildDirectory = "build/client";
const basename = "/";
const future = { "v3_fetcherPersist": true, "v3_relativeSplatPath": true, "v3_throwAbortReason": true, "v3_routeConfig": false, "v3_singleFetch": true, "v3_lazyRouteDiscovery": true, "unstable_optimizeDeps": false };
const isSpaMode = false;
const publicPath = "/";
const entry = { module: entryServer };
const routes = {
  "root": {
    id: "root",
    parentId: void 0,
    path: "",
    index: void 0,
    caseSensitive: void 0,
    module: route0
  },
  "routes/_index": {
    id: "routes/_index",
    parentId: "root",
    path: void 0,
    index: true,
    caseSensitive: void 0,
    module: route1
  }
};
export {
  serverManifest as assets,
  assetsBuildDirectory,
  basename,
  entry,
  future,
  isSpaMode,
  mode,
  publicPath,
  routes
};
