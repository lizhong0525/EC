const { contextBridge, ipcRenderer } = require("electron");

function listen(channel, callback) {
  const handler = (_event, value) => callback(value);
  ipcRenderer.on(channel, handler);
  return () => ipcRenderer.removeListener(channel, handler);
}
contextBridge.exposeInMainWorld("petDesktop", {
  toggleChat: () => ipcRenderer.send("pet:toggle"),
  hideChat: () => ipcRenderer.send("pet:hide"),
  openSettings: () => ipcRenderer.send("pet:settings"),
  openGames: () => ipcRenderer.send("pet:games"),
  openReport: reportId => ipcRenderer.send("pet:report-print", reportId),
  movePet: (dx, dy, x, y) => ipcRenderer.send("pet:move", { dx, dy, x, y }),
  drag: (active, x, y) => ipcRenderer.send("pet:drag", { active, x, y }),
  configure: settings => ipcRenderer.send("pet:configure", settings),
  activity: state => ipcRenderer.send("pet:activity", state),
  quit: () => ipcRenderer.send("pet:quit"),
  onActivity: callback => listen("pet:activity", callback),
  onSettings: callback => listen("pet:settings", callback),
});
