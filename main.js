const { app, BrowserWindow } = require("electron");
const path = require("path");

function create() {
  const win = new BrowserWindow({
    width: 1100,
    height: 800,
    webPreferences: {
      // ✅ Security best-practice: no Node in renderer
      nodeIntegration: false,
      contextIsolation: true,
      // ✅ Use preload to expose a safe API bridge
      preload: path.join(__dirname, "preload.js"),
    },
  });

  win.loadFile("index.html");
}

app.whenReady().then(() => {
  create();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) create();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
