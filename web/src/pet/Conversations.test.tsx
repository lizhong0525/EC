import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import Conversations from "./Conversations";
import { request } from "./api";

vi.mock("./api", () => ({ request: vi.fn() }));
afterEach(() => { cleanup(); vi.clearAllMocks(); });

it("loads additional pages and searches without putting dialogue into URLs", async () => {
  const first = {
    session_id: "first", title: "旅行", updated_at: "2026-10-26T00:00:00Z",
    created_at: "2026-10-26T00:00:00Z", message_count: 2,
  };
  vi.mocked(request).mockImplementation(async (_path, body) => {
    const { query, offset } = body as { query: string; offset: number };
    return {
      items: query ? [] : [{ ...first, session_id: offset ? "second" : "first", title: offset ? "读书" : "旅行" }],
      total: query ? 0 : 2, has_more: !query && offset === 0,
    } as never;
  });
  const onSelect = vi.fn();
  render(<Conversations selected="first" revision={0} disabled={false}
    onSelect={onSelect} onNew={vi.fn()} onClose={vi.fn()} />);
  fireEvent.click(await screen.findByRole("button", { name: "加载更多" }));
  await screen.findByRole("button", { name: "打开对话：读书" });
  expect(screen.getByRole("button", { name: "打开对话：旅行" })).toHaveAttribute("aria-current", "page");
  fireEvent.click(screen.getByRole("button", { name: "打开对话：读书" }));
  expect(onSelect).toHaveBeenCalledWith("second");
  fireEvent.change(screen.getByRole("textbox", { name: "搜索对话" }), { target: { value: "不存在" } });
  await screen.findByText("没有匹配的对话");
  expect(request).toHaveBeenLastCalledWith("/sessions/search", { query: "不存在", offset: 0, limit: 40 });
});

it("discards late results from an older search", async () => {
  let resolveOld: (value: unknown) => void = () => {};
  vi.mocked(request).mockImplementation(async (_path, body) => {
    if ((body as { query: string }).query === "") {
      return await new Promise<unknown>(resolve => { resolveOld = resolve; }) as never;
    }
    return { items: [], total: 0, has_more: false } as never;
  });
  render(<Conversations selected="" revision={0} disabled={false}
    onSelect={vi.fn()} onNew={vi.fn()} onClose={vi.fn()} />);
  await waitFor(() => expect(request).toHaveBeenCalled());
  fireEvent.change(screen.getByRole("textbox", { name: "搜索对话" }), { target: { value: "新搜索" } });
  await screen.findByText("没有匹配的对话");
  resolveOld({ items: [{
    session_id: "stale", title: "旧搜索结果", updated_at: "2026-10-26T00:00:00Z", message_count: 0,
  }], total: 1, has_more: false });
  await waitFor(() => expect(screen.queryByText("旧搜索结果")).not.toBeInTheDocument());
});

it("renames a conversation and deletes the selected record with confirmation", async () => {
  const item = { session_id: "first", title: "原标题", updated_at: "2026-10-26T00:00:00Z",
    created_at: "2026-10-26T00:00:00Z", message_count: 2 };
  vi.mocked(request).mockResolvedValue({ items: [item], total: 1, has_more: false } as never);
  vi.spyOn(window, "confirm").mockReturnValue(true);
  const onNew = vi.fn();
  render(<Conversations selected="first" revision={0} disabled={false}
    onSelect={vi.fn()} onNew={onNew} onClose={vi.fn()} />);
  fireEvent.click(await screen.findByRole("button", { name: "管理对话：原标题" }));
  fireEvent.click(screen.getByRole("menuitem", { name: "重命名" }));
  fireEvent.change(screen.getByRole("textbox", { name: "对话标题" }), { target: { value: "新的标题" } });
  fireEvent.click(screen.getByRole("button", { name: "保存标题" }));
  await waitFor(() => expect(request).toHaveBeenCalledWith("/sessions/first/title", { title: "新的标题" }, "PUT"));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  fireEvent.click(await screen.findByRole("button", { name: "管理对话：原标题" }));
  fireEvent.click(screen.getByRole("menuitem", { name: "删除本机记录" }));
  await waitFor(() => expect(request).toHaveBeenCalledWith("/sessions/first", undefined, "DELETE"));
  expect(onNew).toHaveBeenCalled();
});
