
const {app,BrowserWindow} = require('electron')
function create(){ const win = new BrowserWindow({width:900,height:700,webPreferences:{nodeIntegration:true}}); win.loadFile('index.html'); }
app.whenReady().then(()=>{ create(); app.on('activate',()=>{ if (BrowserWindow.getAllWindows().length===0) create(); });});
app.on('window-all-closed',()=>{ if(process.platform!=='darwin') app.quit(); });
