from pathlib import Path
from PIL import Image,ImageOps
import win32com.client as com,sys,fitz,json
work=Path(__file__).resolve().parent
base=work.parent.parent/'reports'
names=['静态网页','CSS排版','JS交互','网站设计']
def render(f):
    pdf=fitz.open(f); thumbs=[]
    q=work/('QA-'+f.stem.split('_')[-1]);q.mkdir(exist_ok=True)
    rows=[]
    for n,page in enumerate(pdf,1):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.40,1.40),alpha=False)
        pix.save(q/f'page-{n}.png')
        im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        im.thumbnail((485,405));thumbs.append(im)
        blocks=page.get_text('blocks')
        txt=page.get_text()
        rows.append((n,len(txt),len(page.get_images()),' '.join(txt.split())[-72:]))
    cols=2; m=Image.new('RGB',(cols*495,420*((len(thumbs)+cols-1)//cols)),'#b4b7b4')
    for i,img in enumerate(thumbs):m.paste(img,((i%cols)*495+(495-img.width)//2,(i//cols)*420))
    m.save(q/'all.png')
    print(f.name,rows)
w=com.DispatchEx('Word.Application');w.Visible=False;w.DisplayAlerts=0
try:
    for idx in sys.argv[1:]:
        docpath=base/('学号_姓名_网页设计_'+names[int(idx)-1]+'.docx')
        doc=w.Documents.Open(str(docpath),ConfirmConversions=False,AddToRecentFiles=False,ReadOnly=True)
        doc.Repaginate()
        doc.ExportAsFixedFormat(str(docpath.with_suffix('.pdf')),17)
        doc.Close(False)
        render(docpath.with_suffix('.pdf'))
finally:
    w.Quit()
