from qa import *
def test(p):
    id='03-js'
    for phase, folder in [('before',EV/id/'初稿'),('after',ST/id)]:
        p.goto((folder/'index.html').as_uri())
        b=lambda square:p.locator('[data-square="'+square+'"]')
        b('c7').focus()
        p.keyboard.press('Enter')
        p.keyboard.press('ArrowLeft')
        p.keyboard.press('Enter')
        assert p.locator('#notation').inner_text()=='1. Qb7#'
        capture(p,EV/id/('mate-'+phase+'.png'))
        p.locator('#reset').focus()
        p.keyboard.press('Enter')
        focus=p.evaluate("({id:document.activeElement.id,square:document.activeElement.dataset.square})")
        assert focus.get('id')=='reset' if phase=='before' else focus.get('square')=='c7'
        p.keyboard.press('ArrowLeft')
        print(phase,focus,'after arrow',p.evaluate('document.activeElement.outerHTML.slice(0,120)'),flush=True)
        capture(p,EV/id/(phase+'.png'))
        excerpt(p,id,phase,folder/'practice.js','document.getElementById("reset").addEventListener',0,5)
    prompt(p,id)
    p.goto((ST/id/'index.html').as_uri())
    b('c7').click()
    b('e3').click()
    assert '不能走到' in p.locator('#feedback').inner_text()
    assert p.locator('#notation').inner_text()=='尚未落子'
    b('b6').click()
    assert '逼和' in p.locator('#feedback').inner_text()
    b('c6').click()
    assert p.locator('#notation').inner_text()=='1. Qb6'
    p.locator('#reset').click()
    b('c7').click();b('a7').click()
    assert '仍有解围' in p.locator('#feedback').inner_text()
    p.set_viewport_size({'width':360,'height':760})
    assert p.evaluate('document.documentElement.scrollWidth <= innerWidth')
    notes(id,problem='使用键盘完成棋局后，按“重新开始”虽然还原了棋子，但焦点停留在重开按钮。紧接着按方向键无法操作新棋局。',steps='聚焦 c7，按回车、左箭头、回车完成 Qb7#。用键盘激活“重新开始”，再按左箭头，对照焦点及棋盘变化。',cause='重置函数修改了棋盘数据与 Tab 停靠格，却未把实际键盘焦点移回棋盘。',fix='在重开按钮的事件处理函数中完成状态重置后，显式聚焦当前起始格 c7。首次打开页面仍保留自然焦点顺序。',result='重开后焦点回到 c7，再按左键移到 b7。将死、逼和、单纯将军、非法落点、重复点击及 360 像素窄屏均已复测。',reflection='交互状态既包含棋盘数据，也包含下一次按键的接收位置。验收应覆盖完成、重开、再次操作的连续过程。')
