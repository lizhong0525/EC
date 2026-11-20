from __future__ import annotations
import ctypes, html, json, os, time, sys, importlib, textwrap
from pathlib import Path
from PIL import Image, ImageStat
import win32gui, win32ui, win32con
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work' / 'qa'
EV = ROOT / 'evidence'
ST = ROOT / 'stages'
TEMP = WORK / 'temp'
TEMP.mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(TEMP)
ctypes.windll.user32.SetProcessDPIAware()

def capture(p, dest):
    p.bring_to_front()
    p.wait_for_timeout(380)
    title = p.title()
    handles = []
    def match(hwnd,_):
        if win32gui.GetClassName(hwnd) == 'Chrome_WidgetWin_1' and win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd).startswith(title):
            handles.append(hwnd)
    win32gui.EnumWindows(match, None)
    if len(handles) != 1:
        raise RuntimeError(f'Window count {len(handles)} for {title}')
    hwnd = handles[0]
    win32gui.RedrawWindow(hwnd,None,None,win32con.RDW_INVALIDATE|win32con.RDW_ALLCHILDREN|win32con.RDW_UPDATENOW)
    l,t,r,b = win32gui.GetWindowRect(hwnd)
    width,height = r-l,b-t
    hdc = win32gui.GetWindowDC(hwnd)
    src=win32ui.CreateDCFromHandle(hdc)
    mem=src.CreateCompatibleDC()
    bmp=win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(src,width,height)
    old=mem.SelectObject(bmp)
    try:
        if not ctypes.windll.user32.PrintWindow(hwnd,mem.GetSafeHdc(),2):
            raise RuntimeError('PrintWindow failed')
        img=Image.frombytes('RGB',(width,height),bmp.GetBitmapBits(True),'raw','BGRX',0,1)
        assert max(ImageStat.Stat(img).stddev) > 12
        img.save(dest)
    finally:
        mem.SelectObject(old)
        mem.DeleteDC()
        src.DeleteDC()
        win32gui.ReleaseDC(hwnd,hdc)
        win32gui.DeleteObject(bmp.GetHandle())

def reading(p,id,name,title,body,is_code=False):
    dest = WORK / (id+'-'+name+'.html')
    style = 'body{margin:36px auto;max-width:1000px;padding:0 32px;color:#172a28;background:white;font:21px/1.7 "Microsoft YaHei",sans-serif}h1{font-size:27px;margin:0 0 18px;border-bottom:1px solid #aac1b3;padding-bottom:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f4;padding:16px 20px;border-left:4px solid #357153;font:18px/1.65 Consolas,"Microsoft YaHei",monospace}p{margin:.75em 0}'
    content='<pre>'+html.escape(body)+'</pre>' if is_code else ''.join('<p>'+html.escape(s)+'</p>' for s in body.split('\n') if s.strip())
    dest.write_text('<!doctype html><meta charset=utf-8><title>'+html.escape(title)+'</title><style>'+style+'</style><h1>'+html.escape(title)+'</h1>'+content,encoding='utf8')
    p.goto(dest.as_uri())
    capture(p, EV/id/(name+'.png'))

def prompt(p,id):
    lines=(ROOT/'prompts'/(id+'.md')).read_text(encoding='utf8').splitlines()
    reading(p,id,'prompt',lines[0].lstrip('# ')+' · 实际任务书','\n'.join(lines[1:]).replace('## ',''))
def excerpt(p,id,phase,path,pattern,before=2,after=5):
    lines=path.read_text(encoding='utf8').splitlines()
    i=next(i for i,l in enumerate(lines) if pattern in l)
    body='\n'.join(f'{n+1:>3}   {lines[n]}' for n in range(max(i-before,0),min(i+after+1,len(lines))))
    reading(p,id,phase+'-code',path.name+' · '+('修复前' if phase=='before' else '修复后'),body,True)
def browser(fn):
    with sync_playwright() as pw:
        ctx=pw.chromium.launch_persistent_context(str(WORK/'profile'),channel='chrome',headless=False,no_viewport=True,ignore_default_args=['--enable-automation'],args=['--window-size=1120,820','--window-position=30,30','--disable-gpu','--disable-direct-composition','--disable-features=Translate','--no-first-run','--hide-crash-restore-bubble'])
        try:
            p=ctx.pages[0]
            p.set_default_timeout(6000)
            fn(p)
        finally: ctx.close()
def notes(id,**data):
    (EV/id/'notes.json').write_text(json.dumps(data,ensure_ascii=False,indent=2), encoding='utf8')

if __name__=='__main__':
    browser(getattr(importlib.import_module(sys.argv[1]),sys.argv[2]))
