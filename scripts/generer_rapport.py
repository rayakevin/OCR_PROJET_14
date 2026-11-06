"""Rend le rapport Markdown en PDF A4 et contrôle la limite de vingt pages.

Les marqueurs de page conservent le rythme du rapport. Les tableaux répètent
leur en-tête ; la vérification visuelle reste nécessaire après une modification.
"""

from pathlib import Path
import re, html
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted

root=Path.cwd();dest=root/'output/pdf/rapport_technique_ocr14.pdf'
pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVuBold',italic='DejaVu',boldItalic='DejaVuBold')
styles=getSampleStyleSheet()
for name in ['Normal','BodyText','Heading1','Heading2','Heading3','Title']:
 styles[name].fontName='DejaVu'
styles['BodyText'].fontSize=9;styles['BodyText'].leading=13;styles['BodyText'].spaceAfter=7
for name,size in [('Heading1',22),('Heading2',16),('Heading3',11)]:
 styles[name].fontName='DejaVuBold';styles[name].fontSize=size;styles[name].leading=size*1.25;styles[name].textColor=colors.HexColor('#163349');styles[name].spaceAfter=10
small=ParagraphStyle('table',parent=styles['BodyText'],fontSize=7.5,leading=10,spaceAfter=0)
code=ParagraphStyle('code',fontName='DejaVu',fontSize=7,leading=10,backColor=colors.HexColor('#edf4f7'),borderPadding=7,spaceAfter=10)

def inline(s):
 """Échappe le texte avant de convertir le gras et le code Markdown simple."""
 s=html.escape(s).replace('—','-').replace('–','-').replace('‑','-')
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'`([^`]+)`',r'\1',s)
 return s

text=(root/'reports/rapport_technique_ocr14.md').read_text()
lines=text.splitlines();story=[];i=0
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line=='<!-- page -->':story.append(PageBreak());i+=1;continue
 if line.startswith('```'):
  block=[];i+=1
  while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
  story.append(Preformatted('\n'.join(block),code,maxLineLength=92));i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   r=[x.strip() for x in lines[i].strip().strip('|').split('|')];i+=1
   if all(re.fullmatch(r'[:\- ]+',x) for x in r):continue
   rows.append([Paragraph(inline(x),small) for x in r])
  n=len(rows[0]);width=A4[0]-88
  widths=[width/n]*n
  if n==2:widths=[width*.36,width*.64]
  if n==2 and rows[0][0].text=='Emplacement':widths=[width*.53,width*.47]
  if n==3:widths=[width*.25,width*.39,width*.36]
  t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceaf0')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#c7d6de')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
  story.extend([t,Spacer(1,9)]);continue
 if line.startswith('#'):
  level=len(line)-len(line.lstrip('#'))
  story.append(Paragraph(inline(line[level:].strip()),styles['Heading'+str(min(level,3))]));i+=1;continue
 block=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','<!--')) and not re.match(r'^\d+\. ',lines[i]):
  block.append(lines[i].strip());i+=1
 story.append(Paragraph(inline(' '.join(block)),styles['BodyText']))

def footer(canvas,doc):
 canvas.saveState();canvas.setStrokeColor(colors.HexColor('#bdd0db'));canvas.line(44,38,A4[0]-44,38)
 canvas.setFont('DejaVu',7);canvas.setFillColor(colors.HexColor('#526574'))
 canvas.drawString(44,26,'OCR14 • Rapport technique • 6 novembre 2026')
 canvas.drawRightString(A4[0]-44,26,str(doc.page));canvas.restoreState()
SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=52,title='OCR14 - Rapport technique',author='Projet OCR14').build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
reader=PdfReader(dest)
print('PDF',len(reader.pages),'pages',dest.stat().st_size,'octets')
assert len(reader.pages)<=20
for i,p in enumerate(reader.pages):print(i+1,len(p.extract_text()),p.extract_text().splitlines()[0][:80])
