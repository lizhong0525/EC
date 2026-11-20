/* Optional browser regression. Uses an ephemeral TEST-ONLY annotation server. */
const {spawn} = require("node:child_process");
const fs = require("node:fs/promises");
const path = require("node:path");
const assert = require("node:assert/strict");
const {chromium} = require("playwright");
const root = path.resolve(__dirname, "..");
const output = path.join(root, "artifacts", "research", "ui-qa");
const server = spawn(
  path.join(root, ".research-venv", "Scripts", "python.exe"),
  [path.join(__dirname, "research_ui_fixture.py")],
  {cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"]}
);
let browser;
(async () => {
  let received = "";
  const tokens = await new Promise((resolve, reject) => {
    server.stdout.on("data", chunk => {
      received += chunk;
      if (received.includes("\n")) resolve(JSON.parse(received.split("\n")[0]));
    });
    server.on("exit", code => reject(new Error("TEST fixture exited: " + code)));
  });
  for (let i = 0; i < 50; i++) {
    try {
      if ((await fetch("http://127.0.0.1:8766/health")).ok) break;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  await fs.mkdir(output, {recursive: true});
  browser = await chromium.launch({headless: true, channel: "msedge"});
  const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("http://127.0.0.1:8766");
  assert.equal(await page.locator("#workspace").isVisible(), false);
  await page.locator("#token").fill(tokens["expert-a"]);
  await page.locator("#login button[type=submit]").click();
  await page.locator("#workspace").waitFor({state: "visible"});
  assert.equal(await page.locator("#token").inputValue(), "");
  assert.equal(await page.locator("#comparison").innerText(), "");
  assert.equal(await page.locator("#history .message").count(), 0);
  await page.screenshot({path: path.join(output, "desktop.png"), fullPage: true});
  await page.locator("#risk").selectOption("L1");
  await page.locator("#confidence").fill("0.7");
  await page.locator('input[name=acceptable][value=CLARIFY_SYMPTOMS]').check();
  await page.locator("#preferred").selectOption("CLARIFY_SYMPTOMS");
  await page.locator("#target").selectOption("SCHOOL_SUPPORT");
  await page.locator("#evidence-code").fill("TEST_ONLY");
  await page.locator("#add-evidence").click();
  await page.locator("#reason").fill("TEST FIXTURE, NOT AN EXPERT RESULT");
  const response = page.waitForResponse(r => r.url().endsWith("/judgments"));
  await page.locator("#submit").click();
  assert.equal((await response).status(), 200);
  await page.waitForFunction(() => document.querySelector("#progress").textContent.includes("当前节点 2"));
  assert.equal(await page.locator("#history .message").count(), 1);
  await page.setViewportSize({width: 390, height: 844});
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path: path.join(output, "mobile.png"), fullPage: true});
  await page.locator("#logout").click();
  assert.equal(await page.locator("#workspace").isVisible(), false);
  await page.locator("#token").fill(tokens["expert-b"]);
  await page.locator("#login button[type=submit]").click();
  await page.waitForFunction(() => document.querySelector("#progress").textContent.includes("当前节点 1"));
  assert.equal(await page.locator("#history .message").count(), 0);
  assert.deepEqual(errors, []);
  console.log("PASS: desktop/mobile layout, submit/lock/advance, role isolation; TEST fixtures only.");
})().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
}).finally(async () => {
  if (browser) await browser.close();
  server.kill();
});
