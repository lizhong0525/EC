const { app, BrowserWindow, ipcMain, screen, dialog, shell } = require("electron");
const { spawn } = require("node:child_process");
const { existsSync } = require("node:fs");
const net = require("node:net");
const path = require("node:path");

const root = path.resolve(__dirname, "../..");
let pet, panel, gameWindow, backend, origin;
let quitting = false;
let scale = 1;
let topmost = true;
let dragging = false;
let dragTimer = null;
let activity = "idle";
let configuredSize = null;
let resizingPet = false;
let dragOrigin = null;
const secureWebPreferences = {
  preload: path.join(__dirname, "preload.cjs"),
  contextIsolation: true, nodeIntegration: false, sandbox: true, webSecurity: true,
};

if (!process.argv.includes("--qa") && !app.requestSingleInstanceLock()) app.quit();

async function freePort() {
  const server = net.createServer();
  const port = await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
  await new Promise(resolve => server.close(resolve));
  return port;
}

async function startBackend() {
  const supplied = process.argv.find(arg => arg.startsWith("--url="));
  if (supplied) {
    const url = new URL(supplied.slice(6));
    if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname)) {
      throw new Error("Only a local HTTP backend is allowed.");
    }
    return url.origin;
  }
  const python = path.join(root, ".research-venv", "Scripts", "python.exe");
  if (!existsSync(python)) throw new Error("Missing .research-venv Python runtime.");
  const port = await freePort();
  const url = `http://127.0.0.1:${port}`;
  backend = spawn(python, ["-m", "uvicorn", "ecocompanion.pet.api:app", "--host", "127.0.0.1", "--port", String(port)],
    { cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONPATH: path.join(root, "backend", "src"), PYTHONUTF8: "1" } });
  let exitCode;
  backend.on("exit", code => {
    exitCode = code;
    if (!quitting && pet) {
      dialog.showErrorBox("Local service stopped", "Please restart the desktop companion.");
      app.quit();
    }
  });
  backend.stdout.on("data", () => {});
  backend.stderr.on("data", () => {});
  for (let i = 0; i < 100; i++) {
    if (exitCode !== undefined) throw new Error("The local service could not start.");
    try {
      const response = await fetch(`${url}/healthz`);
      if (response.ok && (await response.json()).app === "desktop-pet-preview") return url;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 150));
  }
  throw new Error("The local service did not become ready.");
}

function harden(win) {
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url === "https://github.com/PC2005-cloud/dsh-pet") void shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (new URL(url).origin !== origin) event.preventDefault();
  });
  win.webContents.on("will-attach-webview", event => event.preventDefault());
}

function positionPanel() {
  if (!pet || !panel) return;
  const body = pet.getBounds();
  const area = screen.getDisplayMatching(body).workArea;
  const current = panel.getBounds();
  const width = Math.min(current.width, area.width);
  const height = Math.min(current.height, area.height);
  let x = body.x - width - 8;
  if (x < area.x) x = body.x + body.width + 8;
  x = Math.max(area.x, Math.min(x, area.x + area.width - width));
  const y = Math.max(area.y, Math.min(body.y + body.height - height, area.y + area.height - height));
  panel.setBounds({ x: Math.round(x), y: Math.round(y), width, height });
}

function showPanel(settings = false) {
  positionPanel();
  panel.show();
  panel.focus();
  if (settings) panel.webContents.send("pet:settings");
}

function authorized(event) {
  return [pet?.webContents, panel?.webContents].includes(event.sender);
}

function updateDragPosition() {
  if (!pet || pet.isDestroyed() || !dragOrigin) return;
  const area = screen.getDisplayMatching(dragOrigin.bounds).workArea;
  const { width, height } = dragOrigin.bounds;
  const x = Math.min(area.x + area.width - width,
    Math.max(area.x, dragOrigin.bounds.x + dragOrigin.x - dragOrigin.startX));
  const y = Math.min(area.y + area.height - height,
    Math.max(area.y, dragOrigin.bounds.y + dragOrigin.y - dragOrigin.startY));
  pet.setPosition(Math.round(x), Math.round(y));
  if (!dragging && panel.isVisible()) positionPanel();
}

function configurePet(settings) {
  scale = Math.max(.7, Math.min(1.4, settings.pet_scale / 100));
  topmost = settings.always_on_top;
  if (pet.isAlwaysOnTop() !== topmost) pet.setAlwaysOnTop(topmost);
  if (panel.isAlwaysOnTop() !== topmost) panel.setAlwaysOnTop(topmost);
  const size = [Math.round(280 * Math.max(1, scale)), Math.round(370 * Math.max(1, scale))];
  if (configuredSize?.join() !== size.join()) {
    const [x, y] = pet.getPosition();
    resizingPet = true;
    pet.setBounds({ x, y, width: size[0], height: size[1] });
    configuredSize = size;
    resizingPet = false;
  }
  positionPanel();
}

app.whenReady().then(async () => {
  origin = await startBackend();
  const area = screen.getPrimaryDisplay().workArea;
  const webPreferences = secureWebPreferences;
  pet = new BrowserWindow({
    width: 280, height: 370, x: area.x + area.width - 310, y: area.y + area.height - 390,
    frame: false, thickFrame: false, transparent: true, resizable: false, alwaysOnTop: true,
    skipTaskbar: true, hasShadow: false, show: false, title: "EcoCompanion Pet",
    backgroundColor: "#00000000", webPreferences,
  });
  panel = new BrowserWindow({
    width: Math.min(760, area.width - 24), height: Math.min(710, area.height - 24), minWidth: 360, minHeight: 460,
    frame: false, transparent: false, resizable: true, alwaysOnTop: true,
    show: false, backgroundColor: "#fcfdfc", title: "EcoCompanion",
    webPreferences,
  });
  harden(pet); harden(panel);
  panel.on("close", event => { if (!quitting) { event.preventDefault(); panel.hide(); } });
  pet.on("closed", () => { if (!quitting) app.quit(); });
  pet.on("resize", () => {
    if (resizingPet || !configuredSize) return;
    const bounds = pet.getBounds();
    if (bounds.width !== configuredSize[0] || bounds.height !== configuredSize[1]) {
      resizingPet = true;
      pet.setBounds({ ...bounds, width: configuredSize[0], height: configuredSize[1] });
      resizingPet = false;
    }
  });
  await Promise.all([pet.loadURL(`${origin}/?surface=pet`), panel.loadURL(`${origin}/?surface=panel`)]);
  try {
    const saved = await (await fetch(`${origin}/api/pet/bootstrap`)).json();
    if (saved.settings) {
      configurePet(saved.settings);
    }
  } catch { /* The renderer will load settings and report a connection error. */ }
  // Keep the large transparent margins click-through; only the visible figure and tools hit-test.
  const hitTest = setInterval(() => {
    if (!pet || pet.isDestroyed()) return;
    const point = screen.getCursorScreenPoint();
    const bounds = pet.getBounds();
    const x = point.x - bounds.x, y = point.y - bounds.y;
    const inside = dragging || (x >= bounds.width / 2 - 105 * scale
      && x <= bounds.width / 2 + 105 * scale && y >= 20 && y <= bounds.height - 5);
    pet.setIgnoreMouseEvents(!inside, { forward: true });
  }, 80);
  pet.on("closed", () => clearInterval(hitTest));
  positionPanel();
  pet.showInactive();
  app.on("second-instance", () => { pet.show(); showPanel(); });
}).catch(error => {
  dialog.showErrorBox("EcoCompanion", error.message);
  app.quit();
});

ipcMain.on("pet:toggle", event => {
  if (!authorized(event)) return;
  if (panel.isVisible()) panel.hide(); else showPanel();
});

ipcMain.on("pet:games", async () => {
  if (!gameWindow || gameWindow.isDestroyed()) {
    gameWindow = new BrowserWindow({
      width: 1120, height: 820, minWidth: 360, minHeight: 600,
      backgroundColor: "#f6f7f5", title: "游戏间",
      webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true, webSecurity: true },
    });
    gameWindow.setMenuBarVisibility(false);
    gameWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    gameWindow.webContents.on("will-navigate", (event, url) => {
      if (new URL(url).origin !== new URL(origin).origin) event.preventDefault();
    });
    await gameWindow.loadURL(`${origin}/games`);
  }
  gameWindow.show(); gameWindow.focus();
});
ipcMain.on("pet:hide", event => { if (authorized(event)) panel.hide(); });
ipcMain.on("pet:settings", event => { if (authorized(event)) showPanel(true); });
ipcMain.on("pet:report-print", (event, reportId) => {
  if (!authorized(event) || !/^[0-9a-f-]{36}$/.test(reportId || "")) return;
  const reportWindow = new BrowserWindow({
    width: 820, height: 900, minWidth: 360, minHeight: 460, show: false,
    backgroundColor: "#ffffff", title: "EcoCompanion Wellbeing Report",
    webPreferences: secureWebPreferences,
  });
  harden(reportWindow);
  reportWindow.removeMenu();
  reportWindow.once("ready-to-show", () => reportWindow.show());
  void reportWindow.loadURL(`${origin}/api/pet/reports/${reportId}/print`).catch(error => {
    dialog.showErrorBox("Report unavailable", error.message);
    reportWindow.destroy();
  });
});
ipcMain.on("pet:quit", event => { if (authorized(event)) app.quit(); });
ipcMain.on("pet:move", (event, value) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win || win !== pet || !Number.isFinite(value?.dx) || !Number.isFinite(value?.dy)) return;
  if (!dragOrigin) {
    // A lost native pointer-capture frame may start the gesture without drag:true.
    dragOrigin = { bounds: win.getBounds(), startX: value.x - value.dx, startY: value.y - value.dy };
    dragging = true;
  }
  dragOrigin.x = value.x;
  dragOrigin.y = value.y;
});
ipcMain.on("pet:drag", (event, value) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win || win !== pet || typeof value?.active !== "boolean") return;
  dragging = value.active;
  dragOrigin = value.active && Number.isFinite(value.x) && Number.isFinite(value.y)
    ? { bounds: win.getBounds(), startX: value.x, startY: value.y, x: value.x, y: value.y }
    : null;
  if (value.active && !dragTimer) {
    dragTimer = setInterval(updateDragPosition, 16);
  } else if (!value.active && dragTimer) {
    updateDragPosition();
    clearInterval(dragTimer);
    dragTimer = null;
    positionPanel();
  }
  win.setIgnoreMouseEvents(false);
  // Reset the origin on the first frame of a fresh pointer-down even if the
  // previous window lost capture before its pointer-up event was delivered.
  if (value.active && !Number.isFinite(value.x)) dragOrigin = null;
});
ipcMain.on("pet:configure", (event, value) => {
  if (!authorized(event) || !Number.isFinite(value?.pet_scale) || typeof value?.always_on_top !== "boolean") return;
  configurePet(value);
});
ipcMain.on("pet:activity", (event, state) => {
  if (!authorized(event) || !["idle", "attentive", "wave", "look", "think", "write", "sleep", "stretch"].includes(state)) return;
  activity = state;
  for (const win of [pet, panel]) {
    if (win?.webContents !== event.sender) win?.webContents.send("pet:activity", activity);
  }
});
app.on("window-all-closed", () => app.quit());
app.on("before-quit", () => {
  quitting = true;
  if (backend && !backend.killed) backend.kill();
});
