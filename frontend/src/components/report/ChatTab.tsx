"use client";

import clsx from "clsx";
import { Bot, Send, User as UserIcon } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { useAuth } from "@/components/AuthProvider";
import { CitationText } from "@/components/report/CitationText";
import { Alert, Button, Spinner } from "@/components/ui";
import type { ChatMessage } from "@/lib/types";

const SUGGESTIONS = [
  "Which datasets are used most often?",
  "What limitations do the papers mention?",
  "Which method reports the strongest results?",
];

function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const titles = new Map(message.citations.map((citation) => [citation.number, citation.title]));
  return (
    <li className={clsx("flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={clsx(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-200" : "bg-brand-600 text-white",
        )}
      >
        {isUser ? <UserIcon className="h-4 w-4" aria-hidden /> : <Bot className="h-4 w-4" aria-hidden />}
      </div>
      <div className={clsx("max-w-[85%] space-y-2", isUser && "text-right")}>
        <div
          className={clsx(
            "inline-block rounded-2xl px-4 py-2.5 text-left text-sm leading-6",
            isUser ? "bg-brand-600 text-white" : "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-100",
          )}
        >
          {isUser ? message.content : <CitationText text={message.content} titles={titles} anchorPrefix={`chat-${message.id}-`} />}
        </div>
        {message.citations.length > 0 && (
          <ol className="space-y-1 text-xs text-slate-500 dark:text-slate-400">
            {message.citations.map((citation) => (
              <li key={citation.chunk_id} id={`chat-${message.id}-${citation.number}`} className="scroll-mt-20">
                <span className="font-medium">[{citation.number}]</span> {citation.title} ·{" "}
                {citation.page > 0 ? `${citation.section}, p. ${citation.page}` : "abstract"} — “{citation.snippet}”
              </li>
            ))}
          </ol>
        )}
      </div>
    </li>
  );
}

/** Follow-up questions answered by RAG over this session's papers. */
export function ChatTab({ sessionId }: { sessionId: string }) {
  const { api } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .chatHistory(sessionId)
      .then((history) => !cancelled && setMessages(history))
      .catch((caught: unknown) => !cancelled && setError(caught instanceof Error ? caught.message : "Could not load the chat"))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [api, sessionId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || sending) return;
    setError(null);
    setSending(true);
    setDraft("");
    const pending: ChatMessage = { id: -Date.now(), role: "user", content: question, citations: [], created_at: new Date().toISOString() };
    setMessages((current) => [...current, pending]);
    try {
      const exchange = await api.ask(sessionId, question);
      setMessages((current) => [...current.filter((message) => message.id !== pending.id), exchange.question, exchange.answer]);
    } catch (caught) {
      setMessages((current) => current.filter((message) => message.id !== pending.id));
      setDraft(question);
      setError(caught instanceof Error ? caught.message : "The question could not be answered");
    } finally {
      setSending(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void send(draft);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send(draft);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-slate-500 dark:text-slate-400">
        Ask follow-up questions. Answers use only the papers in this report and cite them with the same reference numbers.
      </p>
      {loading ? (
        <Spinner label="Loading conversation..." />
      ) : (
        <ul className="space-y-5">
          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}
          {sending && (
            <li className="flex gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-600 text-white">
                <Bot className="h-4 w-4" aria-hidden />
              </div>
              <Spinner label="Searching the papers..." />
            </li>
          )}
        </ul>
      )}
      <div ref={bottom} />
      {messages.length === 0 && !loading && (
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => void send(suggestion)}
              className="rounded-full border border-slate-200 px-3 py-1 text-sm text-slate-600 hover:border-brand-400 hover:text-brand-700 dark:border-slate-700 dark:text-slate-300"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
      {error && <Alert tone="error">{error}</Alert>}
      <form onSubmit={handleSubmit} className="flex items-end gap-2 rounded-xl border border-slate-300 bg-white p-2 dark:border-slate-700 dark:bg-slate-900">
        <label htmlFor="chat-input" className="sr-only">
          Your question
        </label>
        <textarea
          id="chat-input"
          value={draft}
          onChange={(event) => setDraft(event.target.value.slice(0, 2000))}
          onKeyDown={handleKeyDown}
          rows={2}
          placeholder="Ask about the papers... (Enter to send, Shift+Enter for a new line)"
          className="flex-1 resize-none bg-transparent px-2 py-1 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none dark:text-slate-100"
        />
        <Button type="submit" loading={sending} disabled={!draft.trim()} icon={<Send className="h-4 w-4" aria-hidden />}>
          Send
        </Button>
      </form>
    </div>
  );
}
