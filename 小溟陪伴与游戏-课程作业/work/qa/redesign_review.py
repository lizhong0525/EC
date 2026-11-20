from qa import *
from shutil import copy2
src=ST/'01-html'
pub=ROOT.parents[1]/'小溟陪伴与游戏_四份作业交付'/'学号_姓名_网页设计_静态网页'/'作品'
ev=EV/'01-html'
# Ship the same self-contained pages that the review opens over file://.
for name in ('about.html','index.html','base.css','guide.css','home.css','assets/wave.png','assets/fish-2.webp','assets/portrait.png','assets/ATTRIBUTION.md'):
    dest=pub/name
    dest.parent.mkdir(exist_ok=True,parents=True)
    copy2(src/name,dest)
def review(p):
    p.context.set_offline(True)
    errors=[]
    p.on('pageerror',lambda err: errors.append(str(err)))
    for file,shot in [('index.html','home-redesign'),('about.html','guide-redesign')]:
        p.goto((pub/file).as_uri())
        assert p.locator('img').evaluate_all('els=>els.every(e=>e.complete && e.naturalWidth>0)')
        assert len(p.locator('h1').all())==1
        capture(p,ev/(shot+'.png'))
    p.goto((pub/'index.html').as_uri())
    for name,target in [('chat','chat-guide'),('games','games-guide')]:
        p.locator('#'+target).click()
        assert p.url.endswith('#'+name)
        p.locator('section[aria-labelledby="'+name+'"] a[href^="index.html#"]').click()
        assert p.url.endswith('#'+target)
        if name=='chat':
            capture(p,ev/'after.png')
    excerpt(p,'01-html','after',pub/'about.html','回到首页的聊天指南入口',0,1)
    for width in (320,390,768,1280):
        p.set_viewport_size({'width':width,'height':820})
        for file in ('index.html','about.html'):
            p.goto((pub/file).as_uri())
            assert p.evaluate('document.documentElement.scrollWidth<=innerWidth'),(width,file)
            if width==1280 and file=='index.html':
                p.screenshot(path=str(WORK/'新版首页.png'))
    assert not errors, errors
    print('PASS both published pages, offline assets, return links, four viewport widths')
if __name__=='__main__':
    with sync_playwright() as pw:
        ctx=pw.chromium.launch_persistent_context(str(WORK/'redesign-profile'),channel='chrome',chromium_sandbox=True,headless=False,no_viewport=True,java_script_enabled=False,ignore_default_args=['--enable-automation'],args=['--window-size=1200,875','--window-position=30,30','--no-first-run','--disable-gpu','--disable-direct-composition'])
        try:review(ctx.pages[0])
        finally:ctx.close()
