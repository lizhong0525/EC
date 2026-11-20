const { spawn } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");
const executable = path.join(__dirname, "../node_modules/electron/dist/electron.exe");
if (!existsSync(executable)) {
  console.error("Install the Electron runtime with: node node_modules/electron/install.js");
  process.exit(1);
}
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
const child = spawn(executable, [path.join(__dirname, "main.cjs"), ...process.argv.slice(2)], {
  cwd: path.resolve(__dirname, "../.."), stdio: "inherit", windowsHide: true, env,
});
child.on("exit", code => { process.exitCode = code ?? 1; });
child.on("error", error => { console.error(error.message); process.exitCode = 1; });
