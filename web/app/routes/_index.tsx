import { useState, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Status = "idle" | "streaming" | "done" | "error";

export default function Index() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!query.trim() || status === "streaming") return;

      // Cancel any in-flight request
      abortRef.current?.abort();
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
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const reader = response.body!.getReader();
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
                bottomRef.current?.scrollIntoView({ behavior: "smooth" });
              }
            } catch {
              // ignore malformed chunks
            }
          }
        }

        setStatus("done");
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        setErrorMsg(err instanceof Error ? err.message : "Unknown error");
        setStatus("error");
      }
    },
    [query, status]
  );

  const handleStop = () => {
    abortRef.current?.abort();
    setStatus("done");
  };

  return (
    <div className="min-h-full flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="text-2xl">✝</div>
          <div>
            <h1 className="text-xl font-semibold text-white">Tabor Research</h1>
            <p className="text-xs text-gray-500">
              Catholic SRI · Stocks · ETFs
            </p>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex flex-col max-w-4xl mx-auto w-full px-6 py-8 gap-6">
        {/* Query form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <label
            htmlFor="query"
            className="text-sm font-medium text-gray-400 uppercase tracking-wide"
          >
            Research Query
          </label>
          <div className="flex gap-3">
            <input
              id="query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder='e.g. "Analyze SPY ETF for Catholic SRI compliance"'
              disabled={status === "streaming"}
              className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50 transition"
            />
            {status === "streaming" ? (
              <button
                type="button"
                onClick={handleStop}
                className="px-5 py-3 bg-red-700 hover:bg-red-600 text-white rounded-lg font-medium transition"
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!query.trim()}
                className="px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg font-medium transition"
              >
                Research
              </button>
            )}
          </div>
        </form>

        {/* Status indicator */}
        {status === "streaming" && (
          <div className="flex items-center gap-2 text-sm text-blue-400">
            <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            Researching…
          </div>
        )}

        {/* Error */}
        {status === "error" && errorMsg && (
          <div className="rounded-lg bg-red-950 border border-red-800 px-4 py-3 text-red-300 text-sm">
            {errorMsg}
          </div>
        )}

        {/* Result */}
        {result && (
          <article className="flex-1 bg-gray-900 border border-gray-800 rounded-xl p-6 prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{result}</ReactMarkdown>
          </article>
        )}

        <div ref={bottomRef} />
      </main>
    </div>
  );
}
