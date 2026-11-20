from pathlib import Path
import base64, os
from io import BytesIO
from PIL import Image
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]
tmp=root/'work/qa/temp'
os.environ['TEMP']=os.environ['TMP']=str(tmp)
asset=root/'../web/public/pet/wave.webm'
data=base64.b64encode(asset.read_bytes()).decode()
with sync_playwright() as pw:
    browser=pw.chromium.launch(channel='chrome',chromium_sandbox=True)
    pg=browser.new_page()
    png=pg.evaluate("""async (src) => {
        const v=document.createElement('video');
        v.muted=true;
        const loaded=new Promise((r,j)=>{v.onloadeddata=r;v.onerror=j;});
        v.src='data:video/webm;base64,'+src;await loaded;
        const seek=new Promise(r=>{v.onseeked=r;});v.currentTime=.6;await seek;
        const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;
        c.getContext('2d').drawImage(v,0,0);return c.toDataURL('image/png').split(',')[1];
    }""",data)
    browser.close()
im=Image.open(BytesIO(base64.b64decode(png))).convert('RGBA')
box=im.getchannel('A').point(lambda a:255 if a>15 else 0).getbbox()
print(im.size,box)
im.crop(box).save(root/'stages/01-html/assets/wave.png')
for n in (2,4,5):
    fish=Image.open(root/f'../web/public/games/fish-{n:02}.png').convert('RGB')
    fish.thumbnail((240,240),Image.Resampling.LANCZOS)
    fish.save(root/f'stages/01-html/assets/fish-{n}.webp',quality=89)
