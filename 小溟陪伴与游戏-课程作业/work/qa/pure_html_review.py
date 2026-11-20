from pathlib import Path
import json
from bs4 import BeautifulSoup
from qa import ROOT, WORK, EV, ST, capture, prompt, excerpt
from playwright.sync_api import sync_playwright

stage=ST/'01-html'
ev=EV/'01-html'
assert sorted(x.name for x in stage.iterdir()) == ['about.html','index.html']
for path in stage.glob('*.html'):
    doc=BeautifulSoup(path.read_text(encoding='utf8'),'html.parser')
    assert not doc.select('script, style, svg, link, [style]')
    assert all(not any(key.startswith('on') for key in el.attrs) for el in doc.find_all(True))
    assert all(im.get('src','').startswith('data:image/webp;base64,') for im in doc.select('img'))
with sync_playwright() as pw:
    ctx=pw.chromium.launch_persistent_context(str(WORK/'pure-html-profile'),channel='chrome',chromium_sandbox=True,headless=False,no_viewport=True,ignore_default_args=['--enable-automation'],args=['--window-size=1180,850','--window-position=35,35','--disable-gpu','--disable-direct-composition','--disable-features=Translate','--no-first-run','--hide-crash-restore-bubble'])
    ctx.set_offline(True)
    p=ctx.pages[0]
    home=(stage/'index.html').as_uri()
    guide=(stage/'about.html').as_uri()
    p.goto(home)
    capture(p,ev/'home-redesign.png')
    p.goto(guide)
    capture(p,ev/'guide-redesign.png')
    for k in ['chat','games']:
        p.goto(home)
        p.locator('#'+k+'-guide').click()
        assert p.url.endswith('#'+k)
        p.locator('section[aria-labelledby='+k+'] a[href="index.html#'+k+'-guide"]').click()
        assert p.url.endswith('#'+k+'-guide')
        assert p.evaluate('document.activeElement.id')==k+'-guide'
        if k=='chat': capture(p,ev/'after.png')
    p.goto(home)
    detail=p.locator('details').first
    assert not detail.evaluate('e=>e.open')
    detail.locator('summary').press('Enter')
    assert detail.evaluate('e=>e.open')
    excerpt(p,'01-html','after',stage/'about.html','href="index.html#chat-guide"',3,3)
    prompt(p,'01-html')
    for w in [320,375,760,1280]:
        p.set_viewport_size({'width':w,'height':825})
        for pg in [home,guide]:
            p.goto(pg)
            assert p.locator('img').evaluate('i => i.complete && i.naturalWidth>0')
            metrics=p.evaluate('[document.documentElement.clientWidth,document.documentElement.scrollWidth,document.images[0].clientWidth]')
            print(w,metrics)
            assert metrics[0]==metrics[1],metrics
    ctx.close()
