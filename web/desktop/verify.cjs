/* Isolated UI QA. No real keys, model downloads or cloud inference. */
const { chromium, _electron } = require("playwright");
const { spawn } = require("node:child_process");
const fs = require("node:fs/promises");
const { existsSync } = require("node:fs");
const path = require("node:path");
const net = require("node:net");
const os = require("node:os");
const assert = require("node:assert/strict");

const root = path.resolve(__dirname, "../..");
const output = path.join(root, "artifacts/pet-preview/qa");
let server, browser, electron;

async function freePort() {
  const socket = net.createServer();
  await new Promise(resolve => socket.listen(0, "127.0.0.1", resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  return port;
}

(async () => {
  const home = await fs.mkdtemp(path.join(os.tmpdir(), "ecopet-qa-"));
  await fs.mkdir(output, { recursive: true });
  const port = await freePort();
  const url = `http://127.0.0.1:${port}`;
  const shim = path.join(home, "provider-shim.py");
  await fs.writeFile(shim, `import httpx
import uvicorn
from ecocompanion.pet import deepseek, pipeline
original = deepseek.create_client
def handler(request):
    return httpx.Response(200, json={"id": "qa-deepseek", "model": "deepseek-flash", "choices": [{"index": 0, "message": {"role": "assistant", "content": "我在听，今天想聊点什么？"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}})
def create_client(config, key, transport=None):
    return original(config, key, transport=httpx.MockTransport(handler))
deepseek.create_client = create_client
pipeline.create_client = create_client
uvicorn.run("ecocompanion.pet.api:app", host="127.0.0.1", port=${port})
`, "utf8");
  server = spawn(path.join(root, ".research-venv/Scripts/python.exe"),
    [shim],
    { cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONPATH: `${path.join(root, "backend/src")}${path.delimiter}${root}`, ECO_PET_HOME: home } });
  server.stderr.on("data", data => console.error("QA-SERVER", String(data)));
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error(`QA server exited: ${server.exitCode}`);
    try { if ((await fetch(`${url}/healthz`)).ok) break; } catch {}
    await new Promise(resolve => setTimeout(resolve, 150));
  }
  const bootstrap = await (await fetch(`${url}/api/pet/bootstrap`)).json();
  const settings = { ...bootstrap.settings, qwen_enabled: false, cloud_consent: true };
  await fetch(`${url}/api/pet/settings`, {
    method: "PUT", headers: { "Content-Type": "application/json", "X-Pet-Token": bootstrap.token },
    body: JSON.stringify({ settings, deepseek_api_key: "qa-test-only" }),
  });
  const executable = [
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
  ].find(existsSync);
  browser = await chromium.launch({ executablePath: executable, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("response", response => {
    if (response.url().includes("/api/pet/") && response.status() >= 400)
      errors.push(`HTTP ${response.status()} ${new URL(response.url()).pathname}`);
  });
  const ready = page.waitForResponse(response => response.url().endsWith("/api/pet/bootstrap") && response.ok());
  await page.goto(url);
  try { await ready; } catch (error) {
    await page.screenshot({ path: path.join(output, "startup-failure.png") });
    console.log(errors);
    throw error;
  }
  await page.getByRole("textbox", { name: "消息", exact: true }).waitFor();
  // Headless Chromium can defer autoplay while initializing a new window.
  await page.locator("video").evaluate(video => video.play());
  await page.waitForFunction(() => {
    const video = document.querySelector("video");
    return video?.readyState >= 2 && !video.paused && video.currentTime > 0;
  });
  const pixels = await page.evaluate(async () => {
    const video = document.querySelector("video");
    const canvas = document.createElement("canvas");
    canvas.width = 640; canvas.height = 360;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    function frame() {
      ctx.clearRect(0, 0, 640, 360); ctx.drawImage(video, 0, 0, 640, 360);
      const data = ctx.getImageData(0, 0, 640, 360).data;
      let opaque = 0, transparent = 0;
      for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 20) opaque++; else transparent++;
      }
      return { opaque, transparent, image: canvas.toDataURL() };
    }
    const first = frame();
    await new Promise(resolve => setTimeout(resolve, 700));
    const second = frame();
    return { opaque: first.opaque, transparent: first.transparent, moving: first.image !== second.image };
  });
  assert(pixels.opaque > 10000, JSON.stringify(pixels));
  assert(pixels.transparent > 100000, "WebM alpha is missing");
  assert(pixels.moving, "Animation is not moving");
  await page.screenshot({ path: path.join(output, "desktop.png") });
  await page.getByRole("textbox", { name: "消息", exact: true }).fill("今天工作有一点累");
  await page.getByRole("button", { name: "发送消息", exact: true }).click();
  await page.getByText("DeepSeek · 已审查", { exact: true }).waitFor();
  const firstSession = await page.evaluate(() => localStorage.getItem("ecocompanion.pet.session"));
  await page.getByRole("textbox", { name: "消息", exact: true }).fill("留在第一组的草稿");
  await page.getByRole("button", { name: "新建对话", exact: true }).click();
  await page.getByRole("textbox", { name: "消息", exact: true }).fill("今天读了一本太空小说");
  await page.getByRole("button", { name: "发送消息", exact: true }).click();
  await page.getByText("DeepSeek · 已审查", { exact: true }).waitFor();
  await page.getByRole("button", { name: "打开对话：今天工作有一点累", exact: true }).click();
  await page.getByText("今天工作有一点累", { exact: true }).waitFor();
  await page.getByRole("alert").filter({ hasText: "上一条消息已完成" }).waitFor({ state: "detached" });
  await page.waitForFunction(() => !document.querySelector(".pet-messages")?.getAttribute("aria-busy").match(/true/));
  assert.equal(await page.locator(".pet-bubble").getByText("今天读了一本太空小说", { exact: true }).count(), 0);
  assert.equal(await page.getByRole("textbox", { name: "消息", exact: true }).inputValue(), "留在第一组的草稿");
  await page.getByRole("textbox", { name: "消息", exact: true }).fill("");
  await page.waitForTimeout(500);
  await page.getByRole("button", { name: "管理对话：今天工作有一点累", exact: true }).click();
  await page.getByRole("menuitem", { name: "重命名", exact: true }).click();
  await page.getByRole("textbox", { name: "对话标题", exact: true }).fill("工作安排");
  await page.getByRole("button", { name: "保存标题", exact: true }).click();
  await page.getByRole("button", { name: "打开对话：工作安排", exact: true }).waitFor();
  for (let i = 0; i < 42; i++) {
    const response = await fetch(`${url}/api/pet/sessions`, {
      method: "POST", headers: { "Content-Type": "application/json", "X-Pet-Token": bootstrap.token },
      body: "{}",
    });
    assert.equal(response.status, 201);
  }
  await fetch(`${url}/api/pet/sessions/${firstSession}/activate`, {
    method: "POST", headers: { "Content-Type": "application/json", "X-Pet-Token": bootstrap.token },
    body: "{}",
  });
  await page.getByRole("button", { name: "关闭对话列表", exact: true }).click();
  await page.getByRole("button", { name: "对话列表", exact: true }).click();
  await page.getByRole("button", { name: "加载更多", exact: true }).click();
  await page.waitForFunction(() => document.querySelectorAll(".pet-history-select[data-session-id]").length === 44);
  await page.getByRole("textbox", { name: "搜索对话", exact: true }).fill("工作");
  await page.getByRole("button", { name: "打开对话：工作安排", exact: true }).waitFor();
  await page.waitForFunction(() => document.querySelectorAll(".pet-history-select[data-session-id]").length === 1);
  await page.getByRole("textbox", { name: "搜索对话", exact: true }).fill("");
  await page.getByRole("button", { name: "加载更多", exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, "conversations.png") });
  await page.getByRole("button", { name: "收起对话", exact: true }).click();
  await page.getByRole("button", { name: "点击桌宠展开对话", exact: true }).click();
  assert(await page.getByRole("textbox", { name: "消息", exact: true }).isVisible());
  await page.getByRole("button", { name: "应用设置", exact: true }).first().click();
  await page.getByRole("button", { name: "人格与表达", exact: true }).click();
  await page.getByLabel("桌宠名字", { exact: true }).fill("测试小溟");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await page.getByText("设置已保存", { exact: true }).waitFor();
  for (const tab of ["评估与决策", "消息渠道"]) {
    await page.getByRole("button", { name: tab, exact: true }).click();
  }
  await page.getByRole("switch", { name: "我确认使用隔离测试账号与测试聊天", exact: true }).check();
  await page.route("**/api/pet/messaging/wechat-contacts", async route => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({
      account: { wxid: "wxid_test_bot", name: "测试机器人" },
      contacts: [{ wxid: "wxid_test_friend", name: "测试好友" }],
    }) });
  });
  await page.getByRole("button", { name: "读取账号与全部好友", exact: true }).click();
  await page.getByText("测试机器人", { exact: true }).waitFor();
  await page.getByRole("checkbox", { name: "测试好友", exact: true }).check();
  await page.getByText("已允许 1 位测试好友", { exact: true }).waitFor();
  await page.getByLabel("搜索测试好友", { exact: true }).fill("测试好友");
  await page.getByLabel("搜索测试好友", { exact: true }).press("Enter");
  await page.getByText("找到 1 个私聊联系人，请选择测试好友", { exact: true }).waitFor();
  assert(await page.getByRole("button", { name: "消息渠道", exact: true }).getAttribute("aria-current") === "page");
  assert.equal(await page.locator(".pet-settings form").count(), 0, "Search must not create a nested form");
  await page.getByRole("button", { name: "刷新全部好友", exact: true }).click();
  await page.getByText("已读取 1 个私聊联系人，请选择测试好友", { exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, "wechat-contact-picker.png") });
  await page.getByRole("switch", { name: "我确认使用隔离测试账号与测试聊天", exact: true }).uncheck();
  await page.getByRole("button", { name: "评估与决策", exact: true }).click();
  const model = page.getByLabel("固定模型版本", { exact: true });
  assert(await model.evaluate(el => el.checkValidity()));
  await page.getByRole("button", { name: "对话服务", exact: true }).click();
  await page.getByRole("textbox", { name: "DeepSeek API Key", exact: true }).fill("sk-test-only-never-sent");
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await page.getByText("设置已保存", { exact: true }).waitFor();
  assert.equal(await page.getByLabel("DeepSeek API Key", { exact: true }).inputValue(), "");
  assert(!(await page.evaluate(() => JSON.stringify(localStorage))).includes("sk-test"));
  await page.screenshot({ path: path.join(output, "settings.png") });
  for (const viewport of [{ width: 390, height: 844 }, { width: 360, height: 640 }]) {
    await page.setViewportSize(viewport);
    for (const tab of ["对话服务", "人格与表达", "评估与决策", "消息渠道", "桌宠与偏好", "隐私与来源"]) {
      await page.getByRole("button", { name: tab, exact: true }).click();
      const modal = await page.getByRole("dialog").boundingBox();
      assert(modal.x >= 0 && modal.x + modal.width <= viewport.width, "Dialog outside viewport");
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      const section = page.locator(".pet-settings-content");
      assert(await section.evaluate(el => el.scrollWidth <= el.clientWidth + 1), `Overflow in ${tab}`);
    }
    await page.screenshot({ path: path.join(output, `mobile-settings-${viewport.width}.png`) });
  }
  await page.getByRole("button", { name: "关闭设置", exact: true }).click();
  await page.getByRole("button", { name: "自我记录与报告", exact: true }).click();
  await page.getByRole("textbox", { name: "今日备注", exact: true }).fill("今天愿意早点休息");
  await page.getByRole("checkbox", { name: "今天工作有一点累", exact: true }).check();
  await page.getByRole("button", { name: "生成预览", exact: true }).click();
  await page.getByRole("heading", { name: "报告预览", exact: true }).waitFor();
  await page.getByRole("button", { name: "保存快照", exact: true }).click();
  await page.getByText("报告已保存到本机", { exact: true }).waitFor();
  await page.locator(".pet-report-item > button").first().click();
  await page.getByRole("heading", { name: "已保存报告", exact: true }).waitFor();
  const printPage = await browser.newPage();
  await printPage.goto(`${url}${await page.getByRole("link", { name: "打开打印 / PDF" }).getAttribute("href")}`);
  await printPage.getByText("今天愿意早点休息", { exact: true }).waitFor();
  await printPage.screenshot({ path: path.join(output, "report-print.png") });
  await printPage.close();
  await page.screenshot({ path: path.join(output, "report-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: path.join(output, "report-mobile.png") });
  await page.getByRole("button", { name: "关闭报告", exact: true }).click();
  if (await page.getByRole("button", { name: "关闭对话列表", exact: true }).count())
    await page.getByRole("button", { name: "关闭对话列表", exact: true }).click();
  await page.getByRole("button", { name: "对话列表", exact: true }).click();
  await page.getByRole("complementary", { name: "历史对话", exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, "mobile-conversations.png") });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  const scrim = await page.locator(".pet-history-scrim").boundingBox();
  await page.mouse.click(scrim.x + scrim.width - 10, scrim.y + 30);
  await page.getByRole("button", { name: "关闭对话列表", exact: true }).waitFor({ state: "detached" });
  await page.screenshot({ path: path.join(output, "mobile-chat.png") });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.reload();
  await page.getByRole("heading", { name: "测试小溟", exact: true }).waitFor();
  assert((await page.locator(".pet-message").count()) >= 3);
  assert.deepEqual(errors, []);
  if (process.argv.includes("--native")) {
    electron = await _electron.launch({
      executablePath: path.join(root, "web/node_modules/electron/dist/electron.exe"),
      args: [path.join(__dirname, "main.cjs"), "--qa", `--url=${url}`],
      cwd: root, env: { ...process.env, ECO_PET_HOME: home },
    });
    let petPage;
    for (let i = 0; i < 60; i++) {
      petPage = electron.windows().find(win => win.url().includes("surface=pet"));
      if (petPage) break;
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    assert(petPage, "Native pet window missing");
    petPage.on("console", message => console.log("PET-CONSOLE", message.text()));
    await petPage.getByRole("button", { name: "点击桌宠展开对话", exact: true }).waitFor();
    await petPage.getByRole("button", { name: "与桌宠对话", exact: true }).click();
    const panel = electron.windows().find(win => win.url().includes("surface=panel"));
    await panel.getByRole("textbox", { name: "消息", exact: true }).waitFor();
    assert(await electron.evaluate(async ({ BrowserWindow }) => {
      for (let i = 0; i < 30; i++) {
        const win = BrowserWindow.getAllWindows().find(item =>
          item.webContents.getURL().includes("surface=panel"));
        if (win?.isVisible()) return true;
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return false;
    }));
    await panel.screenshot({ path: path.join(output, "native-chat.png") });
    await petPage.screenshot({ path: path.join(output, "native-pet.png") });
    await panel.getByRole("button", { name: "应用设置", exact: true }).click();
    await panel.getByRole("dialog", { name: "应用设置", exact: true }).waitFor();
    await panel.screenshot({ path: path.join(output, "native-settings.png") });
    await panel.getByRole("button", { name: "关闭设置", exact: true }).click();
    await panel.waitForTimeout(1000);
    await panel.bringToFront();
    await panel.getByRole("textbox", { name: "消息", exact: true }).waitFor();
    await electron.evaluate(({ BrowserWindow }) => {
      const pet = BrowserWindow.getAllWindows().find(item => item.webContents.getURL().includes("surface=pet"));
      pet.setPosition(700, 400);
    });
    await petPage.waitForTimeout(1500);
    let previousSize = "";
    for (let attempt = 0; attempt < 20; attempt++) {
      const bounds = await electron.evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows().find(item => item.webContents.getURL().includes("surface=pet")).getBounds());
      const size = `${bounds.width}x${bounds.height}`;
      if (size === previousSize) break;
      previousSize = size;
      assert(attempt < 19, `Pet window size did not stabilize: ${JSON.stringify(bounds)}`);
      await petPage.waitForTimeout(100);
    }
    const beforeDrag = await electron.evaluate(({ BrowserWindow }) =>
      BrowserWindow.getAllWindows().find(item => item.webContents.getURL().includes("surface=pet")).getBounds());
    const figure = await petPage.getByRole("button", { name: "点击桌宠展开对话", exact: true }).boundingBox();
    petPage.setDefaultTimeout(15000);
    const pointer = (type, dx, dy, buttons) => petPage.locator(".pet-figure").evaluate((element, data) => {
      element.setPointerCapture = () => {};
      element.dispatchEvent(new PointerEvent(data.type, {
        bubbles: true, cancelable: true, pointerId: 1, pointerType: "mouse", button: 0,
        buttons: data.buttons, clientX: data.x, clientY: data.y,
        screenX: data.bounds.x + data.x, screenY: data.bounds.y + data.y,
      }));
    }, {
      type, bounds: beforeDrag, buttons,
      x: figure.x + figure.width / 2 + dx,
      y: figure.y + figure.height / 2 + dy,
    });
    await pointer("pointerdown", 0, 0, 1);
    for (let index = 1; index <= 12; index++) {
      await pointer("pointermove", index * 10, index * 50 / 12, 1);
      await petPage.waitForTimeout(35);
    }
    await pointer("pointerup", 120, 50, 0);
    await petPage.waitForTimeout(500);
    const afterDrag = await electron.evaluate(({ BrowserWindow }) =>
      BrowserWindow.getAllWindows().find(item => item.webContents.getURL().includes("surface=pet")).getBounds());
    assert(afterDrag.x !== beforeDrag.x || afterDrag.y !== beforeDrag.y, "Native pet did not move");
    assert(Math.abs(afterDrag.x - beforeDrag.x - 120) < 20, `Native drag lost horizontal movement: ${JSON.stringify({ beforeDrag, afterDrag, figure })}`);
    assert(Math.abs(afterDrag.y - beforeDrag.y - 50) < 20, `Native drag lost vertical movement: ${JSON.stringify({ beforeDrag, afterDrag, figure })}`);
    assert.equal(afterDrag.width, beforeDrag.width, "Native drag changed the window width");
    assert.equal(afterDrag.height, beforeDrag.height, "Native drag changed the window height");
    await petPage.screenshot({ path: path.join(output, "native-pet-dragged.png") });
    await panel.getByRole("button", { name: "自我记录与报告", exact: true }).evaluate(element => element.click());
    await panel.locator(".pet-report-item > button").first().evaluate(element => element.click());
    await panel.getByRole("link", { name: "打开打印 / PDF", exact: true }).evaluate(element => element.click());
    await panel.waitForTimeout(500);
    let nativePrint;
    for (let i = 0; i < 30; i++) {
      nativePrint = electron.windows().find(win => win.url().includes("/api/pet/reports/"));
      if (nativePrint) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert(nativePrint, "Native report print window was blocked");
    await nativePrint.getByText("今天愿意早点休息", { exact: true }).waitFor();
    await nativePrint.screenshot({ path: path.join(output, "native-report-print.png") });
    await nativePrint.close();
    await electron.close(); electron = null;
    electron = await _electron.launch({
      executablePath: path.join(root, "web/node_modules/electron/dist/electron.exe"),
      args: [path.join(__dirname, "main.cjs"), "--qa"],
      cwd: root, env: { ...process.env, ECO_PET_HOME: home },
    });
    for (let i = 0; i < 100; i++) {
      const restored = electron.windows().find(win => win.url().includes("surface=panel"));
      if (restored && await restored.getByRole("heading", { name: "测试小溟", exact: true }).count()) {
        assert((await restored.locator(".pet-message").count()) >= 3);
        break;
      }
      assert(i < 99, "Native restart did not restore the session");
      await new Promise(resolve => setTimeout(resolve, 150));
    }
    await electron.close(); electron = null;
  }
  await fs.writeFile(path.join(output, "result.json"),
    JSON.stringify({ passed: true, pixels, pageErrors: errors, conversations: 44, reports: true, native: process.argv.includes("--native") }, null, 2));
  console.log("PASS: animation, chat, 44 conversations, isolation/drafts/search/pagination, settings, secret handling, mobile/native layouts.");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(async () => {
  if (electron) await electron.close();
  if (browser) await browser.close();
  if (server) server.kill();
});
