from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, urllib.request, zipfile, uuid
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
R=Path(__file__).resolve().parents[2]
kinds={'01-html':'静态网页','02-css':'CSS排版','03-js':'JS交互','04-site':'网站设计'}
def ignored(f,root):
    return any(i=='.runtime' or i.startswith('__pycache__') or i.startswith('.') or i in {'work','node_modules'} for i in f.relative_to(root).parts) or f.suffix.lower() in {'.pyc','.tmp','.log','.db'}
def files(root):
    return sorted(x for x in root.rglob('*') if x.is_file() and not ignored(x,root))
def pack(k,title):
    name='学号_姓名_网页设计_'+title
    archive=R/'提交包'/(name+'.zip')
    source=R/'site' if k=='04-site' else R/'stages'/k
    ev=R/'evidence'/k
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files(source):
            z.write(f,'作品/'+f.relative_to(source).as_posix())
        for ext in ['.docx','.pdf']:
            f=R/'reports'/(name+ext)
            assert f.stat().st_size>20000, f
            z.write(f,f.name)
        f=R/'prompts'/(k+'.md')
        z.write(f,'任务书/'+f.name)
        for f in files(ev):
            z.write(f,'过程证据/'+f.relative_to(ev).as_posix())
    dest=R/'work'/'qa'/'unpacked'/str(uuid.uuid4())[:8]/k
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        z.extractall(dest)
    extracted=dest/'作品'
    for f in files(source):
        assert f.read_bytes()==(extracted/f.relative_to(source)).read_bytes()
    if k=='04-site':
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        proc=subprocess.Popen([sys.executable,'-B','server.py','--port',str(port),'--no-browser','--isolated','--profile','zipcheck'],cwd=extracted,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            url=f'http://127.0.0.1:{port}/'
            for _ in range(65):
                if proc.poll() is not None: raise RuntimeError(proc.stderr.read().decode(errors='replace'))
                try:
                    with urllib.request.urlopen(url+'api/state',timeout=1) as h:
                        assert 'settings' in json.load(h)
                    break
                except OSError: time.sleep(.15)
            else: raise RuntimeError('service did not start')
            with urllib.request.urlopen(url) as h: assert b'<html' in h.read().lower()
        finally:
            proc.terminate()
            proc.wait(10)
    else:
        class Links(HTMLParser):
            def handle_starttag(self,tag,attrs):
                for key,val in attrs:
                    if key in {'src','href'} and val:
                        url=urlsplit(val)
                        if not url.scheme and not url.netloc and url.path:
                            assert (self.page.parent/unquote(url.path)).is_file(), (self.page,val)
        for h in extracted.rglob('*.html'):
            p=Links();p.page=h;p.feed(h.read_text(encoding='utf8'))
        assert (extracted/'index.html').is_file()
    print(k,archive.name,archive.stat().st_size,flush=True)
if __name__=='__main__':
    for k,t in kinds.items(): pack(k,t)
