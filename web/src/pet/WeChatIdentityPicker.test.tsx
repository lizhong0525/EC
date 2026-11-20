import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import WeChatIdentityPicker from "./WeChatIdentityPicker";
import { request, type MessagingSettings } from "./api";

vi.mock("./api", () => ({ request: vi.fn() }));
afterEach(() => { cleanup(); vi.clearAllMocks(); });

function settings(changes: Partial<MessagingSettings["personal_wechat"]> = {}) {
  return {
    enabled: false, account_wxid: "", allowed_chats: [], allow_groups: false,
    test_account_confirmed: true, ...changes,
  };
}

it("fills the internal account ID and selects a friend without manual wxid entry", async () => {
  vi.mocked(request).mockResolvedValue({
    account: { wxid: "wxid_test_bot", name: "测试机器人" },
    contacts: [{ wxid: "wxid_test_friend", name: "测试好友" }],
  });
  const value = settings();
  const onChange = vi.fn();
  render(<WeChatIdentityPicker value={value} disabled={false} onChange={onChange} />);
  fireEvent.click(screen.getByRole("button", { name: "读取账号与全部好友" }));
  expect(await screen.findByText("测试机器人")).toBeInTheDocument();
  expect(request).toHaveBeenCalledWith("/messaging/wechat-contacts", {
    confirmed: true, account_wxid: "", query: "",
  });
  expect(onChange).toHaveBeenCalledWith({ account_wxid: "wxid_test_bot" });
  fireEvent.click(screen.getByRole("checkbox", { name: "测试好友" }));
  expect(onChange).toHaveBeenCalledWith({ allowed_chats: ["wxid_test_friend"] });
});

it("clears a previous whitelist when the selected account changes", async () => {
  vi.mocked(request).mockResolvedValue({
    account: { wxid: "wxid_new_bot", name: "新测试账号" },
    contacts: [{ wxid: "wxid_new_friend", name: "新好友" }],
  });
  const onChange = vi.fn();
  render(<WeChatIdentityPicker value={settings({
    account_wxid: "wxid_old_bot", allowed_chats: ["wxid_old_friend"],
  })} disabled={false} onChange={onChange} />);
  fireEvent.click(screen.getByRole("button", { name: "读取账号与全部好友" }));
  await screen.findByText("测试账号已更换，请重新选择好友");
  expect(onChange).toHaveBeenCalledWith({ account_wxid: "wxid_new_bot", allowed_chats: [] });
});

it("requires explicit test-account confirmation before reading contacts", () => {
  const onChange = vi.fn();
  render(<WeChatIdentityPicker value={settings({ test_account_confirmed: false })}
    disabled={false} onChange={onChange} />);
  expect(screen.getByRole("button", { name: "读取账号与全部好友" })).toBeDisabled();
  expect(request).not.toHaveBeenCalled();
});

it("offers an explicit installation action when the optional component is missing", async () => {
  vi.spyOn(window, "confirm").mockReturnValue(true);
  vi.mocked(request)
    .mockRejectedValueOnce(new Error("尚未安装可选依赖 wechatauto-replica"))
    .mockResolvedValueOnce({ ok: true, detail: "桥接组件已安装，可以读取账号与好友" });
  render(<WeChatIdentityPicker value={settings()} disabled={false} onChange={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "读取账号与全部好友" }));
  fireEvent.click(await screen.findByRole("button", { name: "安装桥接组件" }));
  expect(await screen.findByText("桥接组件已安装，可以读取账号与好友")).toBeInTheDocument();
  expect(request).toHaveBeenLastCalledWith("/messaging/wechat-bridge/install", {});
});

it("clears the old search on full refresh and keeps an empty result visible", async () => {
  const allContacts = Array.from({ length: 75 }, (_, index) => ({
    wxid: `wxid_test_${index}`, name: `测试好友${index}`,
  }));
  vi.mocked(request)
    .mockResolvedValueOnce({
      account: { wxid: "wxid_test_bot", name: "测试机器人" }, contacts: allContacts,
    })
    .mockResolvedValueOnce({
      account: { wxid: "wxid_test_bot", name: "测试机器人" }, contacts: [],
    })
    .mockResolvedValueOnce({
      account: { wxid: "wxid_test_bot", name: "测试机器人" }, contacts: allContacts,
    });
  render(<WeChatIdentityPicker value={settings()} disabled={false} onChange={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "读取账号与全部好友" }));
  await screen.findByRole("checkbox", { name: "测试好友74" });
  fireEvent.change(screen.getByLabelText("搜索测试好友"), { target: { value: "不存在" } });
  fireEvent.click(screen.getByRole("button", { name: "搜索好友" }));
  expect(await screen.findByText("没有匹配的好友")).toBeInTheDocument();
  expect(screen.getByText("测试机器人")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "查看全部好友" }));
  await waitFor(() => expect(request).toHaveBeenLastCalledWith("/messaging/wechat-contacts", {
    confirmed: true, account_wxid: "wxid_test_bot", query: "",
  }));
  expect(await screen.findByRole("checkbox", { name: "测试好友74" })).toBeInTheDocument();
  expect(screen.getByLabelText("搜索测试好友")).toHaveValue("");
});

it("searches with Enter without submitting the surrounding settings form", async () => {
  vi.mocked(request).mockResolvedValue({
    account: { wxid: "wxid_test_bot", name: "测试机器人" },
    contacts: [{ wxid: "wxid_test_friend", name: "测试好友" }],
  });
  const saved = vi.fn(event => event.preventDefault());
  render(<form onSubmit={saved}><WeChatIdentityPicker value={settings()}
    disabled={false} onChange={vi.fn()} /></form>);
  fireEvent.click(screen.getByRole("button", { name: "读取账号与全部好友" }));
  await screen.findByRole("checkbox", { name: "测试好友" });
  const input = screen.getByLabelText("搜索测试好友");
  fireEvent.change(input, { target: { value: "测试好友" } });
  fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
  await waitFor(() => expect(request).toHaveBeenLastCalledWith("/messaging/wechat-contacts", {
    confirmed: true, account_wxid: "wxid_test_bot", query: "测试好友",
  }));
  expect(saved).not.toHaveBeenCalled();
  expect(document.querySelector("form form")).toBeNull();
  expect(screen.getByRole("button", { name: "刷新全部好友" })).toBeInTheDocument();
});
