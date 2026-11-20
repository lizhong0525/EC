from __future__ import annotations
import sys
import json
import threading
import uuid
from pathlib import Path
from contextlib import contextmanager
from playwright.sync_api import sync_playwright, expect
from qa import ROOT, WORK, EV, capture

sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "site"))
from server import App, Handler, ThreadingHTTPServer

DEST = EV / "04-site"

@contextmanager
def website():
    data = WORK / "site-test-data" / uuid.uuid4().hex[:10]
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as service:
        service.app = App(data, True)
        worker = threading.Thread(target=service.serve_forever, daemon=True)
        worker.start()
        try:
            yield "http://127.0.0.1:" + str(service.server_port) + "/"
        finally:
            service.shutdown()
            worker.join()

def ready(page):
    expect(page.locator("#startForm button")).to_be_enabled(timeout=20000)

def game(page, kind):
    return page.request.get(page.url.split("#")[0] + "api/state").json()["games"][kind]

def test(page, base, probe):
    if probe:
        page.route("**/app.js", lambda route: route.fulfill(path=WORK / "site-initial-app.js", content_type="text/javascript"))
    page.goto(base + "#home")
    expect(page.get_by_text("第一步 · 设置联网对话")).to_be_visible()
    if not probe:
        capture(page, DEST / "home.png")
        page.get_by_role("link", name="开始对话").click()
        text = page.locator("#chatText")
        send = page.locator("#chatForm button")
        text.fill("今天完成了社团的第一次活动！")
        send.click()
        expect(page.locator("#log .bubble")).to_have_count(2)
        text.fill("下次见面，想约大家一起下棋。")
        def lose_reply(route):
            assert route.fetch().ok
            route.abort("failed")
        page.route("**/api/chat", lose_reply, times=1)
        send.click()
        expect(send).to_have_text("重试发送")
        assert text.input_value() == "下次见面，想约大家一起下棋。"
        send.click()
        expect(page.locator("#log .bubble")).to_have_count(4)
        page.reload()
        expect(page.locator("#log .bubble")).to_have_count(4)
        capture(page, DEST / "chat.png")
    page.goto(base + "#chess")
    color = page.locator('[name="color"]')
    color.select_option("black")
    page.locator("#startForm button").click()
    expect(page.locator(".sq")).to_have_count(64)
    ready(page)
    first = game(page, "chess")
    assert first["human"] == "black" and first["next_actor"] == 0
    assert page.locator(".sq").first.get_attribute("data-square") == "h1"
    if probe:
        assert color.input_value() == "white"
        capture(page, DEST / "before-chess.png")
        print("Original defect reproduced: saved=black, select=white", flush=True)
        return
    assert color.input_value() == "black"
    move = first["legal"][0]
    page.locator(f'[data-square="{move[:2]}"]').click()
    page.locator(f'[data-square="{move[2:4]}"]').click()
    expect(page.locator(".history li")).to_have_count(3)
    ready(page)
    prior = game(page, "chess")
    page.reload()
    ready(page)
    assert game(page, "chess")["fen"] == prior["fen"]
    assert color.input_value() == "black"
    page.locator(f'[data-square="{game(page, "chess")["legal"][0][:2]}"]').click()
    capture(page, DEST / "chess.png")
    page.once("dialog", lambda dialog: dialog.accept())
    page.locator('[data-act="resign"]').click()
    expect(page.locator(".result")).to_contain_text("白方获胜")
    page.goto(base + "#undercover")
    page.locator("#startForm button").click()
    expect(page.locator(".player")).to_have_count(4)
    ready(page)
    spoken = 0
    voted = 0
    for index in range(70):
        ready(page)
        view = game(page, "undercover")
        if view["result"]:
            assert len(view["reveal"]) == 4
            expect(page.get_by_text("身份揭晓")).to_be_visible()
            break
        assert "reveal" not in view and "words" not in view and "votes" not in view
        assert view["next_actor"] == 0
        if index == 1:
            page.reload()
            ready(page)
            assert game(page,"undercover")["revision"] == view["revision"]
            capture(page, DEST / "undercover.png")
        if view["phase"] == "vote":
            page.locator("[data-vote]").first.click()
            voted += 1
        else:
            page.locator("#wordText").fill("这让我想到了一个生活中的场景。" if view["phase"]=="describe" else "我会再比较一下大家刚才的线索。")
            page.locator("#speakForm button").click()
            spoken += 1
        ready(page)
    else:
        raise AssertionError("Word game did not finish")
    assert voted and spoken
    page.reload()
    expect(page.get_by_text("身份揭晓")).to_be_visible()
    print("Passed chat replay, black board refresh, both game completions and delayed disclosure.", flush=True)

def main():
    probe = "--probe" in sys.argv
    DEST.mkdir(parents=True, exist_ok=True)
    with website() as url, sync_playwright() as pw:
        ctx=pw.chromium.launch_persistent_context(str(WORK/"site-profile"), channel="chrome", chromium_sandbox=True, headless=False, no_viewport=True,
            ignore_default_args=["--enable-automation"],args=["--window-size=1150,880", "--window-position=80,45", "--disable-gpu",
            "--disable-direct-composition","--disable-features=Translate","--no-first-run","--hide-crash-restore-bubble"])
        try:
            p=ctx.pages[0]
            p.set_default_timeout(12000)
            test(p, url, probe)
        finally:
            ctx.close()

if __name__=="__main__":
    main()