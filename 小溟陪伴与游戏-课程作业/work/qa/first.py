from qa import *
def html_test(p):
    id='01-html'
    for phase, folder in [('before',EV/id/'初稿'), ('after',ST/id)]:
        p.goto((folder/'index.html').as_uri())
        assert p.locator('img').evaluate("e => e.complete && e.naturalWidth > 0")
        p.get_by_role('link',name='怎样开始聊天').click()
        p.locator('#chat').wait_for()
        p.locator('section[aria-labelledby="chat"] a').click()
        dest=p.evaluate('location.hash')
        print(phase,'return target',dest,flush=True)
        assert dest==('#conversation' if phase=='before' else '#chat-guide')
        capture(p,EV/id/(phase+'.png'))
        excerpt(p,id,phase,folder/'about.html','回到首页的',0,1)
    prompt(p,id)
    notes(id,problem='从首页的聊天指南入口进入说明后，按返回链接会落到上一节的对话示例，使用键盘的读者需要再找一次原来的入口。',steps='双击首页，选择“怎样开始聊天”，在关于页选择本节返回链接，查看地址栏片段及焦点。',cause='初稿把聊天指南的返回目标写成了示例标题 #conversation，两条指南入口本身也没有独立标识。',fix='给首页两条入口链接分别添加 chat-guide、games-guide 标识，并让关于页的返回链接指向发起操作的对应链接。',result='按原路径复测聊天入口落回 #chat-guide，游戏入口落回 #games-guide；离线肖像可加载。',reflection='检查链接时，除了核对目标文件是否存在，还要从实际入口往返一遍，核对键盘焦点能否接续。')
    p.goto((ST/id/'index.html').as_uri())
    p.get_by_role('link',name='怎样进入游戏').click()
    p.locator('section[aria-labelledby="games"] a').click()
    assert p.evaluate('location.hash')=='#games-guide'
def css_test(p):
    id='02-css'
    for phase,folder in [('before',EV/id/'初稿'),('after',ST/id)]:
        p.goto((folder/'index.html').as_uri())
        p.get_by_role('link',name='查看卧底规则').click()
        p.wait_for_timeout(1000)
        marked=p.locator('#word-rules').evaluate("e=>getComputedStyle(e).backgroundColor")
        print(phase,'target background',marked,flush=True)
        assert (marked=='rgba(0, 0, 0, 0)')==(phase=='before')
        capture(p,EV/id/(phase+'.png'))
        excerpt(p,id,phase,folder/'lobby.css','[id] { scroll-margin',0,5)
    prompt(p,id)
    p.goto((ST/id/'index.html').as_uri())
    capture(p,EV/id/'desktop.png')
    for width,cols in [(360,1),(760,2),(1000,2)]:
        p.set_viewport_size({'width':width,'height':740})
        assert len(p.locator('.games').evaluate("e=>getComputedStyle(e).gridTemplateColumns.split(' ')"))==cols
        assert p.evaluate('document.documentElement.scrollWidth <= innerWidth')
    notes(id,problem='桌面点击“查看卧底规则”后，两份规则同时出现在页面底部。用户无法从版面上看出哪一种是刚刚选择的规则。',steps='以桌面窗口打开大厅，选择第二张卡的规则链接；对照当前 URL 的 #word-rules 和两条规则的视觉样式。',cause='两个规则块的外观完全一致。页面到底时，滚动受到剩余高度限制，单靠锚点滚动无法稳定区分目标。',fix='用 CSS :target 为刚跳入的规则添加底色和左边框，并在对应段落上显示焦点轮廓。',result='同一入口再次跳转后，只突出显示卧底规则；切换象棋规则时高亮跟随目标。另实测 360 像素单列、760 与 1000 像素双列，均无整页横向溢出。',reflection='响应式验收需要检查点击后的状态。高度有限的页面在不同视口下可能同时展示多个目标，选中反馈要能够独立表达当前上下文。')
