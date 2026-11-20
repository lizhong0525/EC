import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import PetApp from "./PetApp";
import { bootstrap, request, type SettingsView } from "./api";

vi.mock("./api", () => ({ bootstrap: vi.fn(), request: vi.fn() }));
afterEach(cleanup);

const view: SettingsView = {
  deepseek_key_configured: false, jev_key_configured: false,
  credential_storage: "windows-dpapi", warning: null,
  settings: {
    pet_name: "小溟", personality_prompt: "温和简洁", deepseek_base_url: "https://api.deepseek.com",
    deepseek_model: "deepseek-flash", deepseek_thinking: false, temperature: .7, max_tokens: 512, context_turns: 6,
    timeout_seconds: 35, cloud_consent: false, qwen_enabled: false,
    qwen_base_url: "http://127.0.0.1:11434", qwen_model: "qwen2.5:7b",
    assessment_timeout_seconds: 25, jev_enabled: false, jev_model: "jev-1.14.0",
    jev_confidence: .8, pet_scale: 100, animation_enabled: true, always_on_top: true,
  },
};
beforeEach(() => {
  localStorage.clear();
  vi.mocked(bootstrap).mockResolvedValue(view);
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  HTMLElement.prototype.scrollTo = vi.fn();
  vi.mocked(request).mockImplementation(async path => {
    if (path === "/sessions") return { session_id: "test-session" } as never;
    if (path === "/sessions/search") return { items: [], total: 0, has_more: false } as never;
    if (path.endsWith("/turn")) return {
      result: { response: "我在听，愿意说说今天的事吗？", decision: { used_fallback: true } },
      mode: "safety_fallback", attention: false,
      stages: [{ name: "audit", state: "ok", detail: "审查通过" }],
    } as never;
    return view as never;
  });
});

it("opens a chat from the pet and sends a real API turn", async () => {
  render(<PetApp />);
  await screen.findByRole("heading", { name: "小溟" });
  await waitFor(() => expect(request).toHaveBeenCalledWith("/sessions", {}));
  fireEvent.click(screen.getByRole("button", { name: "收起对话" }));
  expect(screen.queryByRole("textbox", { name: "消息" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "与桌宠对话" }));
  fireEvent.change(screen.getByRole("textbox", { name: "消息" }), { target: { value: "今天有一点累" } });
  fireEvent.click(screen.getByRole("button", { name: "发送消息" }));
  await screen.findByText("我在听，愿意说说今天的事吗？");
  expect(screen.getByText("DeepSeek 不可用 · 安全回退")).toBeInTheDocument();
});

it("saves a personality prompt without persisting API keys in browser storage", async () => {
  render(<PetApp />);
  await screen.findByRole("heading", { name: "小溟" });
  fireEvent.click(screen.getAllByRole("button", { name: "应用设置" })[0]);
  fireEvent.click(screen.getByRole("button", { name: "人格与表达" }));
  fireEvent.change(screen.getByLabelText("人格提示词"), { target: { value: "认真倾听，少问问题" } });
  fireEvent.click(screen.getByRole("button", { name: "保存设置" }));
  await screen.findByText("设置已保存");
  expect(request).toHaveBeenCalledWith("/settings", expect.objectContaining({
    settings: expect.objectContaining({ personality_prompt: "认真倾听，少问问题" }),
    deepseek_api_key: null,
  }), "PUT");
  expect(JSON.stringify(localStorage)).not.toContain("api_key");
});

it("checks real generation and displays a useful provider failure", async () => {
  render(<PetApp />);
  await screen.findByRole("heading", { name: "小溟" });
  fireEvent.click(screen.getAllByRole("button", { name: "应用设置" })[0]);
  expect(screen.getByRole("option", { name: "deepseek-flash" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: "deepseek-v4-pro" })).toBeInTheDocument();
  vi.mocked(request).mockResolvedValueOnce({
    ok: false, detail: "DeepSeek 账户余额不足（HTTP 402），请充值后重试",
  });
  fireEvent.click(screen.getByRole("button", { name: "验证文本生成" }));
  await screen.findByText("DeepSeek 账户余额不足（HTTP 402），请充值后重试");
  expect(request).toHaveBeenCalledWith("/test/deepseek", {});
});

it("switches isolated conversations and restores their unsent drafts", async () => {
  vi.mocked(bootstrap).mockResolvedValue({ ...view, current_session_id: "first" });
  const items = [
    { session_id: "first", title: "海边旅行", created_at: "2026-10-26T00:00:00Z", updated_at: "2026-10-26T00:00:00Z", message_count: 2 },
    { session_id: "second", title: "读书笔记", created_at: "2026-10-26T00:00:00Z", updated_at: "2026-10-26T00:00:00Z", message_count: 2 },
  ];
  vi.mocked(request).mockImplementation(async path => {
    if (path === "/sessions/search") return { items, total: 2, has_more: false } as never;
    if (path.endsWith("/activate")) {
      const id = path.includes("/first/") ? "first" : "second";
      return { session_id: id, attention: false, messages: [{
        message_id: `${id}-message`, role: "USER",
        content: id === "first" ? "旅行对话的内容" : "读书对话的内容",
      }], draft: id === "first" ? "还没发出的旅行草稿" : "" } as never;
    }
    return view as never;
  });
  render(<PetApp />);
  await screen.findByText("旅行对话的内容");
  fireEvent.change(screen.getByRole("textbox", { name: "消息" }), { target: { value: "还没发出的旅行草稿" } });
  fireEvent.click(await screen.findByRole("button", { name: "打开对话：读书笔记" }));
  await screen.findByText("读书对话的内容");
  expect(request).toHaveBeenCalledWith("/sessions/first/draft", { text: "还没发出的旅行草稿" }, "PUT");
  expect(screen.queryByText("旅行对话的内容")).not.toBeInTheDocument();
  expect(screen.getByRole("textbox", { name: "消息" })).toHaveValue("");
  fireEvent.click(screen.getByRole("button", { name: "打开对话：海边旅行" }));
  await screen.findByText("旅行对话的内容");
  expect(screen.queryByText("读书对话的内容")).not.toBeInTheDocument();
  expect(screen.getByRole("textbox", { name: "消息" })).toHaveValue("还没发出的旅行草稿");
  expect(localStorage.getItem("ecocompanion.pet.session")).toBe("first");
});

it("keeps the current conversation when a history selection fails", async () => {
  vi.mocked(bootstrap).mockResolvedValue({ ...view, current_session_id: "first" });
  vi.mocked(request).mockImplementation(async path => {
    if (path === "/sessions/first/activate") return {
      session_id: "first", attention: false,
      messages: [{ message_id: "m1", role: "USER", content: "当前对话" }],
    } as never;
    if (path === "/sessions/second/activate") throw new Error("本地服务连接失败");
    if (path === "/sessions/search") return { total: 1, has_more: false, items: [{
      session_id: "second", title: "另一组", updated_at: "2026-10-26T00:00:00Z", message_count: 0,
    }] } as never;
    return view as never;
  });
  render(<PetApp />);
  await screen.findByText("当前对话");
  fireEvent.click(await screen.findByRole("button", { name: "打开对话：另一组" }));
  await screen.findByText("本地服务连接失败");
  expect(screen.getByText("当前对话")).toBeInTheDocument();
  expect(localStorage.getItem("ecocompanion.pet.session")).toBe("first");
});
