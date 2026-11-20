from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import base64, io, random, math

ROOT=Path(__file__).resolve().parents[2]
ST=ROOT/'stages'/'01-html'
SRC=ROOT/'work'/'01-html-before-pure'/'assets'
RES=Image.Resampling

def illustration(w, h, guide=False):
    rng=random.Random(517 if guide else 129)
    img=Image.new('RGB',(w,h))
    d=ImageDraw.Draw(img)
    def pt(x,y): return round(x*w),round(y*h)
    def poly(points,c): d.polygon([pt(*p) for p in points],fill=c)
    def rect(x1,y1,x2,y2,c): d.rectangle((*pt(x1,y1),*pt(x2,y2)),fill=c)
    def disk(x,y,r,c):
        cx,cy=pt(x,y); rr=round(h*r); d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=c)
    top, bottom = ((41,48,83),(207,141,138)) if not guide else ((49,57,93),(228,167,125))
    for y in range(h):
        t=min(1,(y/h)*1.2)
        color=tuple(round(a+(b-a)*t) for a,b in zip(top,bottom))
        d.line((0,y,w,y),fill=color)
    for i in range(round(w*h*.00033)):
        x,y=rng.randrange(w),rng.randrange(max(1,h*55//100))
        rad=rng.choice([1,1,1,2]); color=rng.choice(['#a8adc1','#cec9bf','#fff0bd']);d.rectangle((x,y,x+rad,y+rad),fill=color)
        if rad==2 and i%5==0:
            d.line((x-3,y+1,x+5,y+1),fill=color);d.line((x+1,y-3,x+1,y+5),fill=color)
    disk(.20,.265,.132,'#f4d096');disk(.206,.26,.103,'#f9dda0')
    for x,y,s in [(.18,.246,.019),(.204,.31,.012),(.22,.268,.015),(.196,.194,.009)]:disk(x,y,s,'#e7bf89')
    for i in range(16):
        x=rng.uniform(0,1); y=rng.uniform(.21,.47); width=rng.uniform(.025,.077)
        rect(x,y,x+width,y+.006,'#b2a1ad')
    for offset, phase, base, col in [(0,.6,.63,'#8b839d'),(0,2.2,.68,'#696d8a'),(0,1.2,.75,'#4a5777')]:
        pts=[(0,.9)]
        for step in range(49):
            x=step/48
            pts.append((x,base-.06*math.sin(x*12+phase)-.024*math.sin(x*33+phase)-rng.uniform(0,.04)))
        pts += [(1,1),(0,1)]
        poly(pts,col)
    rect(0,.71,1,1,'#647489')
    for i in range(round(w*.64)):
        x=rng.randrange(w);y=rng.randrange(round(.72*h),h)
        d.line((x,y,x+rng.randrange(2,max(4,w//40)),y),fill=rng.choice(['#7c8e9a','#9299a3','#606e83']))
    # A small harbour village, with roofs, a beacon, and lit windows.
    for x,y,scale in [(.27,.705,.074),(.34,.708,.057),(.405,.708,.049),(.46,.713,.042)]:
        width=scale*.76
        rect(x-width/2,y-scale,x+width/2,y,'#68657a')
        poly([(x-width*.6,y-scale),(x-.007,y-scale*1.62),(x+width*.6,y-scale)],'#424f6a')
        for shift in [-.19,.19]: rect(x+shift*width,y-.045,x+shift*width+.005,y-.028,'#f1ce98')
    rect(.108,.54,.133,.687,'#e8c9ac')
    rect(.108,.595,.133,.620,'#b56e68')
    rect(.104,.519,.137,.548,'#323e58')
    rect(.109,.492,.131,.521,'#ffdc90')
    poly([(.101,.493),(.12,.461),(.139,.493)],'#512f46')
    poly([(0,.71),(.06,.70),(.104,.674),(.16,.677),(.22,.707),(.30,.715),(0,.74)],'#394c66')
    if not guide:
        poly([(0,.922),(.23,.903),(.53,.92),(.73,.871),(1,.86),(1,1),(0,1)],'#3f505b')
        poly([(0,.971),(.20,.937),(.54,.975),(.69,.92),(1,.916),(1,1),(0,1)],'#293c49')
        for i in range(round(w*.13)):
            x=rng.randrange(w);y=rng.randrange(int(h*.96),h);d.line((x,y,x+2,y-4),fill='#819082')
        char=Image.open(SRC/'wave.png').convert('RGBA')
        ch=int(h*.74);cw=round(ch*char.width/char.height)
        char=char.resize((cw,ch),RES.LANCZOS)
        cx,cy=pt(.73,.94)
        d.ellipse((cx-cw*.4,cy-5,cx+cw*.4,cy+4),fill='#283441')
        img.paste(char,(cx-cw//2,cy-ch),char)
    else:
        poly([(0,.87),(.47,.85),(1,.68),(1,1),(0,1)],'#79594c')
        poly([(.45,.89),(.74,.62),(.965,.747),(.71,1)],'#382f40')
        def square(x,y):
            return (.47 + .282*x + .19*y, .89 - .244*x + .13*y)
        for row in range(8):
            for col in range(8):
                a=col/8;b=row/8
                poly([square(a,b),square(a+.125,b),square(a+.125,b+.125),square(a,b+.125)],'#efe1b8' if (row+col)%2==0 else '#575b75')
        for a,b in [(.29,.30),(.51,.44),(.60,.23),(.82,.57)]:
            x,y=square(a,b); disk(x,y-.027,.023,'#4b3a51')
            poly([(x-.008,y),(x+.008,y),(x+.004,y-.018),(x-.004,y-.018)],'#403749')
    return img

def data(img):
    buf=io.BytesIO();img.save(buf,format='WEBP',quality=82,method=6)
    return 'data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode()

def image_markup(guide=False):
    large=illustration(1120,224 if guide else 290,guide)
    small=illustration(560,200 if guide else 250,guide)
    alt='晚霞里，海岸棋盘上的棋子面向灯塔' if guide else '月亮升起，蓝发的小溟在海湾边的灯塔前挥手'
    return f'''<picture>
        <source media="(max-width: 600px)" sizes="calc(100vw - 32px)" srcset="{data(small)} 560w">
        <img alt="{alt}" sizes="(max-width: 1152px) calc(100vw - 32px), 1120px"
             srcset="{data(large)} 1120w" src="{data(small.resize((280,small.height//2),RES.LANCZOS))}">
      </picture>'''

def doc(title, markup):
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · EcoCompanion</title>
</head>
<body>
{markup}
</body>
</html>
'''

head='''  <header>
    <nav aria-label="站点导航"><p><strong>EcoCompanion</strong> · <a href="index.html">小溟的房间</a> / <a href="about.html">使用手册</a></p></nav>
  </header>'''
home=doc('小溟的房间',head+'''
  <main>
    <h1 id="meet">你好，我是小溟。</h1>
    <p>欢迎来到小溟的房间。在这里，你可以聊一件今天发生的小事，也可以约我下一盘棋。</p>
    '''+image_markup()+'''
    <section aria-labelledby="start">
      <h2 id="start">第一次来，从这里开始</h2>
      <p><a id="chat-guide" href="about.html#chat">怎样开始聊天</a> · <a id="games-guide" href="about.html#games">怎样进入游戏</a> · <a href="about.html#launch">怎样启动完整网站</a></p>
    </section>
    <hr>
    <section aria-labelledby="conversation">
      <h2 id="conversation">一段对话示例</h2>
      <dl>
        <dt><strong>我</strong></dt><dd>社团活动坐了半天，我还是没敢开口。</dd>
        <dt><strong>小溟</strong></dt><dd>第一次找机会开口确实不容易。他们聊到什么时，你最想接一句？</dd>
        <dt><strong>我</strong></dt><dd>有人提到周末的棋局。我也想去看看。</dd>
      </dl>
      <details><summary>我也可以这样开始吗？</summary><p>可以。打开完整网站后，进入「对话」，写一句你愿意分享的事，再按「发送」。<a href="about.html#chat">查看聊天步骤</a></p></details>
    </section>
  </main>
  <footer><hr><p>原作 EcoCompanion · <a href="about.html#credits">角色署名与素材来源</a></p></footer>''')
guide=doc('小溟使用手册',head+'''
  <main>
    <h1>小溟使用手册</h1>
    <nav aria-label="手册目录"><p><a href="#launch">Windows 启动</a> · <a href="#chat">聊天与连接</a> · <a href="#games">游戏玩法</a> · <a href="#credits">素材来源</a></p></nav>
    '''+image_markup(True)+'''
    <section aria-labelledby="launch">
      <h2 id="launch">在 Windows 启动完整网站</h2>
      <ol>
        <li><strong>解压。</strong>准备 Python 3.10 或更新版本，解压完整网站，打开含有 <code>server.py</code> 的文件夹。</li>
        <li><strong>启动。</strong>双击「启动小溟.cmd」，保留启动窗口。浏览器会自动打开；也可复制窗口中显示的本机网址。</li>
        <li><strong>选择模式。</strong>本地规则无需密钥。要让 DeepSeek 回应，先在首页输入自己的 API 密钥并开启联网授权。</li>
      </ol>
      <details><summary>关掉窗口后，怎样继续？</summary><p>从同一目录重新运行启动程序，可以继续此前保存的聊天和游戏。</p></details>
    </section>
    <hr>
    <section aria-labelledby="chat">
      <h2 id="chat">怎样开始聊天</h2>
      <p>在完整网站首页点「开始对话」，选择运行方式。在「写给小溟」里写下今天想分享的事，然后发送。</p>
      <dl>
        <dt><strong>本地规则</strong></dt><dd>按照问候、心情、游戏等话题选择预设回复，断开外部网络也可使用。</dd>
        <dt><strong>DeepSeek 联网</strong></dt><dd>模型结合最近几轮联网对话作答。发送失败时检查网络，在原消息上点「重试发送」。</dd>
      </dl>
      <p><a href="index.html#chat-guide">回到首页的聊天指南入口</a></p>
    </section>
    <section aria-labelledby="games">
      <h2 id="games">怎样进入游戏</h2>
      <details open><summary><strong>国际象棋</strong></summary><p>从游戏间进入棋桌，选择执白或执黑。点选己方棋子，再点合法目标格。棋谱会记录双方走法；将死对方就能取胜。</p></details>
      <details open><summary><strong>谁是卧底</strong></summary><p>你与三位玩家各获一个秘密词。三人同词，一人异词。轮流给线索，避开词语本身，再讨论、投票。平票时辩护复投。平民要找出卧底，卧底则要留到只剩两人。</p></details>
      <p><a href="index.html#games-guide">回到首页的游戏指南入口</a></p>
    </section>
    <section aria-labelledby="credits">
      <h2 id="credits">角色与素材</h2>
      <p>小溟的角色名为溟月，原画师为上善无形，动画来自 dsh-pet 社区。首页挥手姿态取自 EcoCompanion 自带动画。海湾与棋桌场景为本课程绘制，角色素材沿用原作的个人非商业使用条件及署名。<a href="https://github.com/PC2005-cloud/dsh-pet">原动画项目（联网）</a></p>
    </section>
  </main>
  <footer><hr><p><a href="index.html#meet">回到首页认识小溟</a></p></footer>''')
for name,html in [('index.html',home),('about.html',guide)]:
    (ST/name).write_text(html,encoding='utf8')
