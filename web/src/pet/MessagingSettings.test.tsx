import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import MessagingSettingsPanel from "./MessagingSettings";
import { request } from "./api";

vi.mock("./api", () => ({ request: vi.fn() }));
afterEach(() => { cleanup(); vi.clearAllMocks(); });

const view = {
  settings: {
    qq: { enabled: false, app_id: "", intents: 1 << 25 },
    wechat: { enabled: false, app_id: "", allow_ip_cidr: "" },
    personal_wechat: { enabled: false, account_wxid: "", allowed_chats: [], allow_groups: false, test_account_confirmed: false },
    callback_host: "127.0.0.1", callback_port: 8765, per_user_per_minute: 6, max_inflight: 8,
  },
  qq_secret_configured: false, wechat_token_configured: false,
  wechat_secret_configured: false, credential_storage: "windows-dpapi",
  personal_wechat_token_configured: false, deepseek_key_configured: false, cloud_consent: false,
};

it("submits channel credentials without echoing secrets into the settings view", async () => {
  vi.mocked(request).mockImplementation(async path => (["/messaging/runtime", "/messaging/wechat-bridge"].includes(String(path))
    ? { running: false, channels: [], detail: "未启动" } : view) as never);
  render(<MessagingSettingsPanel />);
  fireEvent.change(await screen.findByLabelText("QQ AppID"), { target: { value: "test-app" } });
  fireEvent.change(screen.getByLabelText("QQ AppSecret"), { target: { value: "test-only-secret" } });
  fireEvent.click(screen.getByRole("button", { name: "保存渠道配置" }));
  await screen.findByText("渠道配置已保存；启动网关后生效");
  expect(request).toHaveBeenCalledWith("/messaging/settings", expect.objectContaining({
    qq_client_secret: "test-only-secret",
  }), "PUT");
  await waitFor(() => expect(screen.getByLabelText("QQ AppSecret")).toHaveValue(""));
});

it("keeps personal WeChat isolated and does not start its UI bridge", async () => {
  vi.mocked(request).mockImplementation(async path => (["/messaging/runtime", "/messaging/wechat-bridge"].includes(String(path))
    ? { running: false, channels: [], detail: "未启动" } : view) as never);
  render(<MessagingSettingsPanel />);
  expect(await screen.findByText("默认关闭")).toBeInTheDocument();
  expect(screen.getByText(/不会随应用启动/)).toBeInTheDocument();
  expect(screen.getByRole("switch", { name: "启用个人微信测试桥接" })).not.toBeChecked();
  expect(screen.queryByText("微信公众号 · 文本客服")).not.toBeInTheDocument();
  expect(screen.queryByRole("switch", { name: "启用微信公众号渠道" })).not.toBeInTheDocument();
  expect(screen.queryByLabelText("公众号回调 Token")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("公众号 AppID")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("公众号 AppSecret")).not.toBeInTheDocument();
});

it("disables a legacy Official Account configuration when saving remaining channels", async () => {
  const legacyView = { ...view, settings: { ...view.settings, wechat: {
    ...view.settings.wechat, enabled: true, app_id: "legacy-app",
  } } };
  vi.mocked(request).mockImplementation(async path => (["/messaging/runtime", "/messaging/wechat-bridge"].includes(String(path))
    ? { running: false, channels: [], detail: "未启动" } : legacyView) as never);
  render(<MessagingSettingsPanel />);
  await screen.findByLabelText("QQ AppID");
  fireEvent.click(screen.getByRole("button", { name: "保存渠道配置" }));
  await screen.findByText("渠道配置已保存；启动网关后生效");
  expect(request).toHaveBeenCalledWith("/messaging/settings", expect.objectContaining({
    settings: expect.objectContaining({
      wechat: expect.objectContaining({ enabled: false, app_id: "legacy-app" }),
    }),
  }), "PUT");
});

it("does not create a nested form or navigate when saving", async () => {
  vi.mocked(request).mockImplementation(async path => (["/messaging/runtime", "/messaging/wechat-bridge"].includes(String(path))
    ? { running: false, channels: [], detail: "未启动" } : view) as never);
  const { container } = render(<MessagingSettingsPanel />);
  await screen.findByLabelText("QQ AppID");
  expect(container.querySelector("form")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "保存渠道配置" }));
  await screen.findByText("渠道配置已保存；启动网关后生效");
  expect(request).toHaveBeenCalledWith("/messaging/settings", expect.anything(), "PUT");
});

it("does not describe an initializing bridge as ready", async () => {
  vi.mocked(request).mockImplementation(async path => {
    if (String(path) === "/messaging/runtime") {
      return { running: true, channels: ["personal_wechat"], detail: "网关运行中" } as never;
    }
    if (String(path) === "/messaging/wechat-bridge") {
      return { running: true, ready: false, channels: [], detail: "正在初始化微信测试桥接" } as never;
    }
    return { ...view, settings: { ...view.settings, personal_wechat: {
      ...view.settings.personal_wechat, enabled: true,
    } } } as never;
  });
  render(<MessagingSettingsPanel />);
  expect(await screen.findByText("正在初始化微信测试桥接")).toBeInTheDocument();
  expect(screen.queryByText("监听已就绪")).not.toBeInTheDocument();
});
