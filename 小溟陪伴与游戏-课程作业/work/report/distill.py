from pathlib import Path
from zipfile import ZipFile
import hashlib,json
from lxml import etree as E
P = Path(__file__).resolve().parent
lines = ['# 课程报告模板约定', '', '资料为用户指定的两个 Word 文档。保留封面的校名、校徽、黑色报告名、横线式资料表，以及正文的校名页眉、装订线、横线页脚。']
for idx in (2,3):
    f=P/f'reference{idx}.docx'
    with ZipFile(f) as z:
        parts={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
        (P/f'parts{idx}.json').write_text(json.dumps(parts,indent=2),encoding='utf-8')
        desc='A4 纵向封面，A4 横向正文，源模板 5 页' if idx==2 else 'A4 纵向封面及正文，源模板 3 页'
        lines.extend(['',f'## {f.name}',desc, 'SHA256 '+hashlib.sha256(f.read_bytes()).hexdigest(),f'渲染参见 reference{idx}.png/pdf，包内哈希见 parts{idx}.json。'])
        r=E.fromstring(z.read('word/document.xml'))
        for sec in r.xpath('//w:sectPr',namespaces={'w':r.nsmap['w']}):
            for node in sec:
                if E.QName(node).localname in ('pgSz','pgMar'):
                    lines.append(E.QName(node).localname+' '+str({E.QName(k).localname:v for k,v in node.attrib.items()}))
lines.extend(['','保留 document.xml 正文最初九个节点及最终节属性。封面双列表第 1 栏课程、第 2–5 栏个人信息、第 9 栏日期为填充区域。报告名、印章和左栏保留。删除原示例以及单纯解释提交规则的页面。正文截图按报告可读性单幅放大排版，汉字以宋体五号为主，图说居中。正文技术章节沿用模板编号。各节正文可增页。页脚示例总数改为动态本节页数，当前页沿用模板页码。正文标题黑色，修复说明和反思写入实际取证内容。'])
(P/'artifact.md').write_text('\n'.join(lines),encoding='utf-8')
