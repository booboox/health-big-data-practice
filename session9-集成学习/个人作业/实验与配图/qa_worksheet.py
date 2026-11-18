from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import sys
sys.stdout.reconfigure(encoding='utf-8')
b=Path(__file__).resolve().parent/'.qa_final'
r=PdfReader(str(b/'worksheet.pdf'))
print('Pages:',len(r.pages))
for i,p in enumerate(r.pages,1):
    text=p.extract_text()
    print(i,len(text),text[:90].replace('\n',' '))
images=convert_from_path(str(b/'worksheet.pdf'),dpi=100,poppler_path=r'C:/Users/10840/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
for i,im in enumerate(images,1):im.save(b/f'page-{i}.png')
for start in range(0,len(images),6):
    sheet=Image.new('RGB',(1260,1250),'#d9dde1');d=ImageDraw.Draw(sheet)
    for j,im in enumerate(images[start:start+6]):
        thumb=im.copy();thumb.thumbnail((400,580))
        x=(j%3)*420+10;y=(j//3)*625+30
        sheet.paste(thumb,(x,y));d.text((x,y-22),f'Page {start+j+1}',fill='black')
    sheet.save(b/f'contact-{start//6+1}.png')
