import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const SESSION = {
  session_id: "session-1",
  state: {
    session_id: "session-1",
    phase: "COMPANION",
    risk_level: "L0",
    relationship: {
      trust: 0.4,
      dependency: 0.1,
      offline_help_willingness: 0.6,
    },
    handoff_status: "NONE",
    memory_enabled: true,
    proactive_enabled: false,
    turn_index: 0,
    updated_at: "2026-09-03T00:00:00Z",
  },
  model: null,
  research_mode: true,
  participant_type: "adult_research_tester",
  synthetic_case_id: null,
};

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function requestUrl(input: RequestInfo | URL): string {
  return typeof input === "string" ? input : input.toString();
}

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  vi.unstubAllGlobals();
});

describe("EcoCompanion research console", () => {
  it("restores a session and supports memory deletion plus trace review", async () => {
    window.localStorage.setItem(
      "ecocompanion.research_session_id",
      SESSION.session_id,
    );
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = requestUrl(input);
        if (url.endsWith("/api/v1/sessions/session-1")) {
          return jsonResponse(SESSION);
        }
        if (url.endsWith("/api/v1/sessions/session-1/messages")) {
          return jsonResponse([
            {
              message_id: "user-1",
              ordinal: 0,
              role: "USER",
              content: "这是刷新前的消息",
              simulated_time: "day-01",
              created_at: "2026-09-03T00:00:00Z",
            },
            {
              message_id: "assistant-1",
              ordinal: 1,
              role: "COMPANION",
              content: "我还在这里。",
              simulated_time: "day-01",
              created_at: "2026-09-03T00:00:01Z",
            },
          ]);
        }
        if (url.endsWith("/api/v1/sessions/session-1/memory")) {
          return jsonResponse([
            {
              memory_id: "memory-1",
              session_id: "session-1",
              kind: "PREFERENCE",
              content: "请叫我林同学",
              confidence: 1,
              created_at: "2026-09-03T00:00:00Z",
              expires_at: null,
              user_authorized: true,
            },
          ]);
        }
        if (url.endsWith("/api/v1/research/sessions/session-1/trace")) {
          return jsonResponse([
            {
              trace_id: "trace-1",
              step_id: 0,
              selected_action: "SUPPORTIVE_LISTENING",
              response_text: "我还在这里。",
              safety_audit: {
                passed: true,
                violation_codes: [],
                used_fallback: false,
              },
            },
          ]);
        }
        if (
          url.endsWith("/api/v1/sessions/session-1/memory/memory-1") &&
          init?.method === "DELETE"
        ) {
          return new Response(null, { status: 204 });
        }
        throw new Error(`unexpected fetch: ${init?.method ?? "GET"} ${url}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(await screen.findByText("这是刷新前的消息")).toBeInTheDocument();
    expect(screen.getByText("我还在这里。")).toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: "显示研究面板" }),
    );
    expect(await screen.findByText("请叫我林同学")).toBeInTheDocument();
    expect(screen.getByText("SUPPORTIVE_LISTENING")).toBeInTheDocument();
    expect(screen.getByText("AUDIT PASS")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "删除" }));
    await waitFor(() => {
      expect(screen.queryByText("请叫我林同学")).not.toBeInTheDocument();
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/sessions/session-1/memory/memory-1",
      { method: "DELETE" },
    );
  });

  it("buffers and displays an audited SSE turn", async () => {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = requestUrl(input);
        if (
          url.endsWith("/api/v1/sessions") &&
          init?.method === "POST"
        ) {
          return jsonResponse(SESSION, 201);
        }
        if (
          url.endsWith("/api/v1/sessions/session-1/turn") &&
          init?.method === "POST"
        ) {
          const updatedSession = {
            ...SESSION,
            state: {
              ...SESSION.state,
              phase: "ASSESS",
              risk_level: "L2",
              turn_index: 1,
            },
          };
          const result = {
            session: updatedSession,
            response: "谢谢你告诉我，我们一起确认一下安全情况。",
            decision: {
              primary_action: "DIRECT_SAFETY_CHECK",
              escalation_level: "L2",
              escalation_target: "TRUSTED_ADULT",
              risk_estimate: {
                depression_worsening: 0.7,
                immediate_safety_risk: 0.55,
                dependency_risk: 0.1,
              },
              uncertainty: 0.2,
              evidence_codes: ["SLEEP_CHANGE"],
              reason_codes: ["SAFETY_CLARIFICATION_REQUIRED"],
              safety_passed: true,
              used_fallback: false,
              audit_codes: [],
              model_calls: [
                {
                  model: "qwen-local-test",
                  input_tokens: 120,
                  output_tokens: 24,
                  cached_input_tokens: 0,
                  estimated_cost_cny: 0,
                  retry_index: 0,
                  cache_hit: false,
                },
              ],
              policy_version: "web-conversation-policy-v1.0.0",
              prompt_version: "web-response-renderer-v2.0.0",
            },
          };
          const encoder = new TextEncoder();
          const body = new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(
                encoder.encode(
                  'event: token\ndata: {"token":"谢谢你告诉我，"}\n\n',
                ),
              );
              controller.enqueue(
                encoder.encode(
                  `event: done\ndata: ${JSON.stringify({ result })}\n\n`,
                ),
              );
              controller.close();
            },
          });
          return new Response(body, {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          });
        }
        throw new Error(`unexpected fetch: ${init?.method ?? "GET"} ${url}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();

    render(<App />);

    const composer = await screen.findByPlaceholderText(
      "输入合成测试对话或成年研究测试内容…",
    );
    await waitFor(() => expect(composer).toBeEnabled());
    await user.type(composer, "我最近总是睡不着");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(
      await screen.findByText("谢谢你告诉我，我们一起确认一下安全情况。"),
    ).toBeInTheDocument();
    expect(screen.getByText("DIRECT_SAFETY_CHECK")).toBeInTheDocument();
    expect(screen.getByText("SLEEP_CHANGE")).toBeInTheDocument();
    expect(screen.getByText("70%")).toBeInTheDocument();
    expect(screen.getByText("本轮已审核模型回复 · 1 次实际调用")).toBeInTheDocument();
    expect(screen.getByText(/qwen-local-test.*24 tokens/)).toBeInTheDocument();
  });

  it("loads a generated benchmark report without hiding safety scope", async () => {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = requestUrl(input);
        if (
          url.endsWith("/api/v1/sessions") &&
          init?.method === "POST"
        ) {
          return jsonResponse(SESSION, 201);
        }
        if (url.endsWith("/api/v1/research/benchmark/runs")) {
          return jsonResponse({
            schema_version: "1.0",
            ignored_directory_names: [],
            runs: [
              {
                schema_version: "1.0",
                run_id: "track-a-paper-run",
                directory_name: "track-a-paper-run",
                track: "A",
                created_at: "2026-09-03T00:00:00Z",
                status: "complete",
                completed_unit_count: 24,
                planned_unit_count: 24,
                methods: ["eco-safe"],
                report_available: true,
              },
            ],
          });
        }
        if (
          url.endsWith(
            "/api/v1/research/benchmark/runs/track-a-paper-run/report",
          )
        ) {
          return jsonResponse({
            schema_version: "1.0",
            run: {
              schema_version: "1.0",
              run_id: "track-a-paper-run",
              directory_name: "track-a-paper-run",
              track: "A",
              created_at: "2026-09-03T00:00:00Z",
              status: "complete",
              completed_unit_count: 24,
              planned_unit_count: 24,
              methods: ["eco-safe"],
              report_available: true,
            },
            report: {
              schema_version: "1.0",
              run_id: "track-a-paper-run",
              completed_unit_count: 24,
              planned_unit_count: 24,
              complete: true,
              rows: [
                {
                  policy_name: "eco-safe",
                  depression_brier: 0.1234,
                  prohibited_action_rate: 0,
                },
              ],
            },
          });
        }
        throw new Error(`unexpected fetch: ${init?.method ?? "GET"} ${url}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(
      screen.getByRole("button", { name: "Benchmark" }),
    );
    expect(
      await screen.findByRole("heading", { name: "track-a-paper-run" }),
    ).toBeInTheDocument();
    expect(screen.getByText("depression_brier")).toBeInTheDocument();
    expect(screen.getByText("0.1234")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Markdown 报告" })).toHaveAttribute(
      "href", "/api/v1/research/benchmark/runs/track-a-paper-run/download?format=md",
    );
    expect(
      screen.getByText(/不构成临床诊断.*治疗效果/),
    ).toBeInTheDocument();
  });
});
