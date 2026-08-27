"""Generate the medical_build course PDF with ReportLab.

Usage:
    python generate_course_pdf.py [output.pdf]
"""
from __future__ import annotations
from pathlib import Path
import re
import sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"docs/medical_build/MEDICAL_BUILD_COURSE.md"
DEFAULT=ROOT/"docs/medical_build/MEDICAL_BUILD_COURSE.pdf"

def inline(text: str) -> str:
    text=re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text=re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text=re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"<link href='\2' color='blue'>\1</link>", text)
    return text

def parse_table(lines, i):
    rows=[]
    while i<len(lines) and lines[i].strip().startswith("|"):
        cells=[c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not all(set(c)<=set("-:") for c in cells): rows.append(cells)
        i+=1
    return rows,i

def build_story(source: Path):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CourseTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=25, spaceAfter=12))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=15, leading=19, textColor=colors.HexColor("#163a5f"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#285f8f"), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.3, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.5, leading=9))
    story=[]; lines=source.read_text(encoding="utf-8").splitlines(); i=0; in_code=False; code=[]
    while i<len(lines):
        line=lines[i]
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code), styles["Code"])); code=[]; in_code=False
            else: in_code=True
            i+=1; continue
        if in_code: code.append(line); i+=1; continue
        if not line.strip(): story.append(Spacer(1,2)); i+=1; continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["CourseTitle"])); i+=1; continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["H1x"])); i+=1; continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["H2x"])); i+=1; continue
        if line.startswith("|"):
            rows,i=parse_table(lines,i)
            if rows:
                table=Table([[Paragraph(inline(c),styles["Smallx"]) for c in row] for row in rows], repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceaf7")),("GRID",(0,0),(-1,-1),0.3,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)])); story.append(table); story.append(Spacer(1,6)); continue
        if line.startswith("> "):
            story.append(Paragraph("<i>"+inline(line[2:])+"</i>", styles["Bodyx"])); i+=1; continue
        story.append(Paragraph(inline(line), styles["Bodyx"])); i+=1
    return story

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica",7); canvas.setFillColor(colors.grey); canvas.drawString(18*mm,10*mm,"foampilot — medical_build"); canvas.drawRightString(192*mm,10*mm,f"Page {doc.page}"); canvas.restoreState()

def main():
    output=Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT; output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=16*mm,title="Cours medical_build")
    doc.build(build_story(SOURCE), onFirstPage=footer, onLaterPages=footer)
    print(output)
if __name__=="__main__": main()
