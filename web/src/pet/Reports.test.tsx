import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import Reports from "./Reports";
import { request } from "./api";

vi.mock("./api", () => ({ request: vi.fn() }));

beforeEach(() => {
  vi.mocked(request).mockReset();
  vi.mocked(request).mockImplementation(async (path, body, method) => {
    if (path === "/reports" && method === "GET") return { items: [] } as never;
    if (path === "/reports/sources") return { items: [] } as never;
    if (path === "/reports/preview") return {
      report_id: "r1", preview_id: "p1", created_at: "2026-10-27T00:00:00Z",
      check_in: (body as { check_in: unknown }).check_in, summary: "本次记录中，你提到加班后的疲惫。",
      summary_source: "extractive", evidence: [{ message_id: "m1", content: "今天加班有点累", created_at: "" }], notice: "非诊断",
      assessments: [{ qwen: { status: "ok", wellbeing_level: "strained", rationale: "原话提到疲劳", evidence_quotes: ["加班有点累"] }, decision: { action: "SUPPORTIVE_LISTENING", level: "L1" } }],
    } as never;
    return {} as never;
  });
});

it("previews saved assessment evidence without generating Qwen report wording", async () => {
  render(<Reports sessionId="s1" messages={[{ message_id: "m1", role: "USER", content: "今天加班有点累" }]} onClose={() => {}} />);
  await waitFor(() => expect(request).toHaveBeenCalledWith("/reports", undefined, "GET"));
  fireEvent.click(screen.getByRole("checkbox", { name: "今天加班有点累" }));
  fireEvent.click(screen.getByRole("button", { name: "生成预览" }));
  await screen.findByRole("heading", { name: "报告预览" });
  await screen.findByText("支持需要：轻度困扰");
  expect(request).toHaveBeenCalledWith("/reports/preview", expect.objectContaining({ source: "local", message_ids: ["m1"], summary: "" }));
  expect(screen.queryByRole("switch")).toBeNull();
  fireEvent.change(screen.getByPlaceholderText("写下你希望保留的回顾；不填写则引用所选原话"), { target: { value: "我修订后的回顾" } });
  expect(screen.getByRole("button", { name: "保存快照" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "保存快照" }));
  await screen.findByText("报告已保存到本机");
  expect(request).toHaveBeenCalledWith("/reports", expect.objectContaining({ summary: "我修订后的回顾", preview_id: "p1" }));
});
