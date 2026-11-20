"""Build the four course reports from the retained Word templates and real observation logs."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from io import BytesIO
import json,sys
from lxml import etree as ET
from docx import Document
from docx.shared import Pt, Mm
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image,ImageDraw,ImageFont

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent
DATA=json.loads((WORK/'content.json').read_text(encoding='utf-8-sig'))
REPORTS=ROOT/'reports'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS={'w':W,'r':R}
PREFIX='学号_姓名_网页设计_'

def paragraph(doc,text='',kind='text',label=''):
    p=doc.add_paragraph()
    f=p.paragraph_format
    f.space_before=Pt(0)
    f.space_after=Pt({'title':6,'h2':7,'caption':8,'text':5,'figure':2}[kind])
    f.line_spacing=1.25 if kind!='figure' else 1.0
    f.widow_control=False
    f.keep_with_next=False
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER if kind in ('title','caption','figure') else WD_ALIGN_PARAGRAPH.LEFT
    fonts={'title':('黑体',16),'h2':('黑体',13),'text':('宋体',10.5),'caption':('宋体',9)}
    def add(s,b):
        if not s:return
        font,size=fonts[kind]
        run=p.add_run(s);run.bold=b
        run.font.name='Times New Roman' if kind=='text' else font
        run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),font)
        run.font.size=Pt(size)
    if label:add(label+'　',True)
    add(text,kind in ('title','h2'))
    snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');p._p.get_or_add_pPr().append(snap)
    return p

def figure(doc,image,caption,max_width,max_height):
    with Image.open(image) as im:w,h=im.size
    actual=min(467, max_width, (350 if max_width>=500 else max_height)*w/h)
    p=paragraph(doc,kind='figure')
    drawing=p.add_run().add_picture(str(image),width=Pt(actual));drawing._inline.docPr.set('descr',caption)
    paragraph(doc,caption,'caption')

def page(doc,title):
    p=paragraph(doc,title,'h2');p.paragraph_format.page_break_before=True

def form_template(idx,name):
    src=WORK/f'reference{2 if idx<4 else 3}.docx'
    doc=Document()
    sec=doc.sections[0]
    sec.page_width=Mm(210)
    sec.page_height=Mm(297)
    sec.left_margin=sec.right_margin=Mm(22)
    sec.top_margin=Mm(22)
    sec.bottom_margin=Mm(20)
    sec.gutter=Mm(0)
    sec.footer_distance=Mm(10)
    sec.different_first_page_header_footer=True
    normal=doc.styles['Normal']
    normal.font.name='Times New Roman'
    normal.font.size=Pt(10.5)
    normal._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    doc.core_properties.author=''
    doc.core_properties.last_modified_by=''
    doc.core_properties.title='网页制作教程 · '+name
    footer=sec.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple')
    field.set(qn('w:instr'),' PAGE ')
    footer._p.append(field)
    with ZipFile(src) as ref:
        for n in (2,3):
            p=doc.add_paragraph()
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            if n==2:p.paragraph_format.space_before=Pt(25)
            p.paragraph_format.space_after=Pt(17)
            p.add_run().add_picture(BytesIO(ref.read(f'word/media/image{n}.png')),width=Mm(64 if n==2 else 29))
            if n==2:
                title=paragraph(doc,'课程设计报告','title')
                title.runs[0].font.size=Pt(30)
                title.paragraph_format.space_after=Pt(17)
    spacer=doc.add_paragraph()
    spacer.paragraph_format.space_after=Pt(19)
    fields=[('课程','网页制作教程：'+name),('学院(系)',''),('专业',''),('姓名',''),('学号',''),('提交时间','2026.11.19')]
    table=doc.add_table(rows=0,cols=2)
    table.alignment=WD_TABLE_ALIGNMENT.CENTER
    table.autofit=False
    table.columns[0].width=Mm(31)
    table.columns[1].width=Mm(96)
    for label,value in fields:
        row=table.add_row()
        for j,(cell,text) in enumerate(zip(row.cells,[label,value])):
            cell.width=Mm(31 if j==0 else 96)
            p=cell.paragraphs[0]
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before=Pt(9)
            p.paragraph_format.space_after=Pt(5)
            run=p.add_run(text)
            run.font.size=Pt(13)
            if j:
                borders=OxmlElement('w:tcBorders')
                line=OxmlElement('w:bottom')
                for key,setting in [('val','single'),('sz','4'),('color','7D7D7D')]:line.set(qn('w:'+key),setting)
                borders.append(line)
                cell._tc.get_or_add_tcPr().append(borders)
    doc.add_page_break()
    return src,doc

def save(src,doc,name):
    dest=REPORTS/(PREFIX+name+'.docx')
    doc.save(dest)
    return dest

def capture(n):
    return json.loads(n.read_text(encoding='utf-8-sig'))

def small(idx):
    s=DATA['small'][idx-1]; ev=ROOT/'evidence'/s['key']; note=capture(ev/'notes.json')
    src,d=form_template(idx,s['name']);paragraph(d,f'{idx}、{s["title"]}','title')
    paragraph(d,f'{idx}.1  智能编程结果','h2')
    for label,val in s['design']:paragraph(d,val,label=label)
    figure(d,ev/'prompt.png',f'图 1　{subj(idx)}部分的指示性提示词',520,185)
    page(d,'初版网页与操作路径')
    paragraph(d,note['steps'],label='操作步骤')
    figure(d,ev/'before.png','图 2　初版网页的实际浏览器画面',545,260)
    paragraph(d,note['problem'],label='观察记录')
    page(d,'初版代码与问题定位')
    paragraph(d,note['cause'],label='原因定位')
    figure(d,ev/'before-code.png','图 3　初版源文件中的相关代码',545,285)
    page(d,f'{idx}.2  人工修复情况')
    paragraph(d,note['fix'],label='修改内容')
    figure(d,ev/'after.png','图 4　按同一路径复测后的网页',545,245)
    paragraph(d,note['result'],label='复测结果')
    if idx==1:
        paragraph(d,'首页和指南各自包含完整的 WebP 位图。正文使用原生标题、列表和折叠问答，两个 HTML 文件放在同一目录即可离线阅读。',label='纯 HTML 终稿')
    page(d,'修复代码与反思')
    figure(d,ev/'after-code.png','图 5　对应修复的代码',545,215)
    paragraph(d,f'{idx}.3  反思与感想','h2')
    paragraph(d,note['reflection'],label='本次收获')
    paragraph(d,s['review'],label='后续检查')
    if idx==1:
        page(d,'纯 HTML 终稿  角色首页')
        paragraph(d,'海湾场景作为一幅位图嵌在首页。读者能阅读紧邻插画的角色介绍，选择聊天、游戏或启动入口，也能展开对话示例后的常见问题。',label='阅读路径')
        figure(d,ev/'home-redesign.png','图 6　纯 HTML 首页的海湾插画与使用入口',545,350)
        page(d,'纯 HTML 终稿  新手指南')
        paragraph(d,'指南从海岸棋盘插画开始。目录可直达 Windows 启动、聊天连接、游戏玩法与素材来源。启动步骤使用有序列表，聊天和游戏章节各自提供返回入口。',label='页面设计')
        figure(d,ev/'guide-redesign.png','图 7　纯 HTML 指南的海岸棋盘插画与章节目录',545,350)
    if idx==2:
        page(d,'附图  游戏大厅的宽屏布局')
        paragraph(d,'两张入口卡在同一行各自展示一种游戏。棋盘和头像分组将人数、游戏目标与下方规则联系起来。')
        figure(d,ev/'desktop.png','附图 A　桌面视口的双栏大厅',545,278)
    if idx==3:
        page(d,'附图  一步将死的实测结果')
        paragraph(d,'白王位于 c6、白后位于 c7，黑王位于 a8。白后走到 b7 后受到白王保护，并控制 a7 和 b8，黑王无法解除将军。棋谱用 Qb7# 记录这一着。')
        figure(d,ev/'mate-after.png','附图 A　键盘走出 Qb7# 后的棋盘与状态反馈',545,270)
    print(save(src,d,s['name']))

def subj(i):return ('HTML','CSS','JavaScript')[i-1]

def architecture():
    f=WORK/'architecture.png'
    img=Image.new('RGB',(1600,665),'white');a=ImageDraw.Draw(img)
    regular='C:/Windows/Fonts/msyh.ttc';bold='C:/Windows/Fonts/msyhbd.ttc'
    def t(x,y,s,size=32,b=False,c='#1f3631'):
        a.text((x,y),s,font=ImageFont.truetype(bold if b else regular,size),fill=c,anchor='mm')
    def box(x,y,w,h,label,sub='',fill='#f3f7f4'):
        a.rounded_rectangle((x,y,x+w,y+h),radius=13,fill=fill,outline='#587c70',width=3)
        t(x+w/2,y+h*(.36 if sub else .5),label,33,True)
        if sub:t(x+w/2,y+h*.73,sub,25)
    def arrow(x1,y1,x2,y2):
        a.line((x1,y1,x2,y2),fill='#517268',width=4)
        a.polygon([(x2-8,y2-12),(x2+8,y2-12),(x2,y2)],fill='#517268')
    xs=[60,560,1060]
    for x,h,s in zip(xs,['HTML · 角色与指南','CSS · 游戏大厅','JS · 离线练习'],['阅读顺序、页面导航','卡片布局、屏幕适配','选子、落子、局面反馈']):
        box(x,24,480,108,h,s);arrow(x+240,132,x+240,182)
    box(60,183,1480,103,'完整网站 · 首页与连接设置','浏览器显示对话和游戏状态，接收输入并向本地服务发送操作','#e0eee6')
    for x in xs:arrow(x+240,286,x+240,345)
    for x,h,s in zip(xs,['小溟对话','国际象棋','谁是卧底'],['连续交流与等待反馈','合法走法、棋谱与结果','私有词语、发言与投票']):
        box(x,346,480,106,h,s);arrow(x+240,452,x+240,517)
    box(60,518,1480,108,'本地 Python 服务','维护会话与游戏进度；获得授权后连接 DeepSeek','#e0eee6')
    img.save(f)
    return f

def final_site():
    idx=4;name='网站设计';ev=ROOT/'evidence'/'04-site'
    src,d=form_template(idx,name)
    paragraph(d,'4、设计网站与说明','title');paragraph(d,'4.1  图例及简介','h2')
    paragraph(d,'小溟陪伴与游戏把聊天、国际象棋和「谁是卧底」三个活动安排在一个本地站点内。首页提供连接设置，并保留一项可离线使用的棋局练习。',label='网站内容')
    figure(d,ev/'home.png','图 6　完整网站首页',411,295)
    paragraph(d,'页面沿用小溟肖像和鱼形头像，通过相同的角色形象串联对话与游戏。棋盘用明暗格区分位置，发言和提示使用文本表达。',label='美学设计')
    paragraph(d,'聊天允许用户选择愿意谈论的事，猜词规则则要求倾听他人并依据公开线索投票。友善交流、尊重意愿和公平参与都有具体的操作场景。',label='价值观契合')
    page(d,'网站框架与使用路径')
    figure(d,architecture(),'图 7　课程四阶段关联与完整网站总体框架',418,190)
    paragraph(d,'网页端接收消息、显示棋盘和玩家发言。本地服务校验操作，维护会话及局面，在用户开启联网后调用模型。棋局练习可单独打开。',label='技术实现')
    figure(d,ev/'chat.png','图 8　小溟对话界面（以页面所示连接状态为准）',393,218)
    page(d,'对局与游戏')
    figure(d,ev/'chess.png','图 9　国际象棋对局页面',395,225)
    figure(d,ev/'undercover.png','图 10　谁是卧底的玩家视图',395,225)
    for key,title in [('need','4.2  需求分析与意义'),('feasible','4.3  技术可行性')]:
        page(d,title)
        for label,val in DATA[key]:paragraph(d,val,label=label)
    note=capture(ev/'notes.json')
    page(d,'4.4  调试与修复')
    for label,field in [('问题现象','problem'),('复现路径','steps'),('原因分析','cause'),('修复方式','fix'),('复测结果','result')]:
        paragraph(d,note[field],label=label)
    paragraph(d,'测试使用单独新建的会话。原配置的密钥与云端授权状态通过检测后，测试从本地 /api/chat 请求 DeepSeek 并收到了非空回答。演示模式下的截图按界面显示的连接状态解读。',label='真实云端验证')
    page(d,'修复前后的棋桌')
    figure(d,ev/'before-chess.png','图 11　初版创建黑方棋桌后，执棋颜色与当前棋桌不符',413,256)
    paragraph(d,'图 9 展示修复后的棋桌。测试从修改颜色这一入口开始，先核对棋盘方向、选择控件和当前玩家，再刷新页面并验证重开的颜色。',label='对照方法')
    paragraph(d,'4.5  反思与感想','h2')
    paragraph(d,note['reflection'])
    paragraph(d,'从首页链接、规则高亮到棋盘落子，每次作业都要求沿着读者实际会走的路径检查。完整站点又增加了等待和刷新这些时间上的变化。把界面截图同操作记录放在一起，才容易看出读者是否获得了当时需要的反馈。')
    print(save(src,d,name))

if __name__=='__main__':
    for a in sys.argv[1:]:
        if a=='4':final_site()
        else:small(int(a))
