"""Build and publish the three course source archives and the full delivery ZIP."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from tempfile import TemporaryDirectory
from io import BytesIO
from uuid import uuid4
import os, shutil
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
ECO=ROOT.parents[1]
GROUP='小溟陪伴与游戏_四份作业交付'
PUB=ECO/GROUP
COURSES=[
    ('01-html','静态网页',('index.html','about.html')),
    ('02-css','CSS排版',('index.html','lobby.css')),
    ('03-js','JS交互',('index.html','practice.css','position.js','practice.js')),
]
GUIDE=('前三份压缩包只包含可运行的课程源码。解压后进入“作品”，双击 index.html。\r\n'
       '第一份为纯 HTML；第二份为 HTML 和 CSS；第三份为 HTML、JavaScript 和棋盘 CSS。\r\n'
       '前三份的可编辑 Word 报告、PDF、实际任务书、截图与调试记录均在“课程材料”中按作业排列。\r\n'
       '第四份保留完整网站、报告与过程材料。解压后双击“作品/启动小溟.cmd”，需要 Python 3.10 或更新版。\r\n'
       '封面个人信息留白，可在提交前填写并更新 PDF。\r\n').encode('utf-8-sig')

def archive(items):
    buf=BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for rel, data in items.items():z.writestr(rel,data)
    result=buf.getvalue()
    with ZipFile(BytesIO(result)) as z:assert z.testzip() is None
    return result

def write(path,data):
    if path.is_file() and path.read_bytes()==data:return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.publishing-'+uuid4().hex)
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)

def materials(stage,name):
    out={}
    for ext in ['.docx','.pdf']:
        f=ROOT/'reports'/(name+ext);out[f.name]=f.read_bytes()
    f=ROOT/'prompts'/(stage+'.md');out['任务书/'+f.name]=f.read_bytes()
    evidence=ROOT/'evidence'/stage
    for f in sorted(evidence.rglob('*')):
        if f.is_file():out['过程证据/'+f.relative_to(evidence).as_posix()]=f.read_bytes()
    return out

def test(browser,where,stage):
    ctx=browser.new_context(offline=True,java_script_enabled=stage=='03-js',viewport={'width':1120,'height':800})
    p=ctx.new_page()
    p.goto((where/'index.html').as_uri())
    if stage=='01-html':
        for f in ['index.html','about.html']:
            p.goto((where/f).as_uri())
            assert p.locator('script, style, link[rel~="stylesheet"], [style], svg').count()==0
            assert not p.evaluate('Array.from(document.querySelectorAll("*")).some(e=>Array.from(e.attributes).some(a=>a.name.startsWith("on")))')
            assert p.locator('img').evaluate_all('els=>els.every(e=>e.complete && e.naturalWidth>0)')
        for anchor, target in [('chat-guide','chat'),('games-guide','games')]:
            p.goto((where/'index.html').as_uri())
            p.locator('#'+anchor).click()
            assert p.url.endswith('#'+target)
            p.locator('a[href="index.html#'+anchor+'"]').first.click()
            assert p.url.endswith('#'+anchor)
    elif stage=='02-css':
        assert p.locator('script').count()==0
        assert p.locator('.seat').evaluate_all('els=>els.length==4 && els.every(e=>e.complete && e.naturalWidth>0)')
        for width,cols in [(360,1),(1000,2)]:
            p.set_viewport_size({'width':width,'height':800})
            assert len(p.locator('.games').evaluate('e=>getComputedStyle(e).gridTemplateColumns.split(" ")'))==cols
        p.get_by_role('link',name='查看卧底规则').click()
        assert p.locator('#word-rules').evaluate('e=>e.matches(":target")')
    else:
        p.locator('[data-square="c7"]').focus()
        for key in ['Enter','ArrowLeft','Enter']:p.keyboard.press(key)
        assert p.locator('#notation').inner_text()=='1. Qb7#'
        p.locator('#reset').click()
        assert p.locator('#notation').inner_text()=='尚未落子'
    ctx.close()

def main():
    package_dir=ROOT/'提交包'
    outer={'交付说明.txt':GUIDE}
    published={}
    report_files={}
    with TemporaryDirectory(dir=ROOT/'work/qa') as temp, sync_playwright() as engine:
        browser=engine.chromium.launch(channel='chrome',chromium_sandbox=True)
        for stage,title,names in COURSES:
            name='学号_姓名_网页设计_'+title
            items={'作品/'+f:(ROOT/'stages'/stage/f).read_bytes() for f in names}
            archive_bytes=archive(items)
            dest=Path(temp)/stage
            with ZipFile(BytesIO(archive_bytes)) as z:
                assert set(z.namelist())==set(items)
                z.extractall(dest)
            test(browser,dest/'作品',stage)
            published[name]=(items,archive_bytes)
            outer[name+'.zip']=archive_bytes
            docs=materials(stage,name)
            report_files[name]=docs
            for rel,data in docs.items():outer['课程材料/'+name+'/'+rel]=data
            print(name, list(items),flush=True)
        browser.close()
    full='学号_姓名_网页设计_网站设计.zip'
    outer[full]=(package_dir/full).read_bytes()
    with ZipFile(BytesIO(outer[full])) as z:
        assert z.testzip() is None and '作品/启动小溟.cmd' in z.namelist()
    master=archive(outer)
    backup=ROOT/'work/qa'/'previous-delivery'/uuid4().hex
    for name,(items,data) in published.items():
        folder=PUB/name
        assert folder.resolve().parent==PUB.resolve()
        for rel,content in report_files[name].items():
            write(PUB/'课程材料'/name/rel,content)
        # Keep superseded files locally. Only touch artifacts from these three submissions.
        for old in (list(folder.rglob('*')) if folder.exists() else []):
            if not old.is_file():continue
            rel=old.relative_to(folder).as_posix()
            if rel in items:continue
            dst=backup/name/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            old.rename(dst)
        for empty in sorted([p for p in folder.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
            if not any(empty.iterdir()):empty.rmdir()
        for rel,content in items.items():write(folder/rel,content)
        assert {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==set(items)
        write(package_dir/(name+'.zip'),data)
        write(PUB/(name+'.zip'),data)
    write(PUB/full,outer[full])
    write(PUB/'交付说明.txt',GUIDE)
    for folder in (ROOT,ECO):write(folder/(GROUP+'.zip'),master)
    with ZipFile(ECO/(GROUP+'.zip')) as z:
        assert all(z.read(name+'.zip')==(PUB/(name+'.zip')).read_bytes() for name in published)
    print('Published and verified',ECO/(GROUP+'.zip'))
if __name__=='__main__':main()
