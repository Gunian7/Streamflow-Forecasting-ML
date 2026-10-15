from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent.parent

def content(doc):
    return ([p.text for p in doc.paragraphs if p.text.strip()],
            [[c.text for r in t.rows for c in r.cells] for t in doc.tables])

for source in ROOT.glob('*补充版-排版优化版.docx'):
    doc = Document(source)
    before = content(doc)
    images = len(doc.inline_shapes)
    for section in doc.sections:
        section.top_margin = Cm(1.6)
        section.bottom_margin = Cm(1.6)
        section.left_margin = Cm(1.7)
        section.right_margin = Cm(1.7)
    # Remove explicit blank spacer paragraphs and manual page breaks.
    for p in list(doc.paragraphs):
        for br in list(p._p.xpath('.//w:br')):
            if br.get(qn('w:type')) == 'page':
                br.getparent().remove(br)
        if not p.text.strip() and not p._p.xpath('.//w:drawing'):
            p._p.getparent().remove(p._p)
            continue
        fmt=p.paragraph_format
        fmt.page_break_before=False
        fmt.keep_with_next=False
        fmt.keep_together=False
        fmt.space_before=Pt(0)
        fmt.space_after=Pt(5)
        fmt.line_spacing=1.18
        fmt.widow_control=True
        for run in p.runs:
            run.font.name='Calibri'
            run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
            run.font.size=Pt(9.5)
            run.font.color.rgb=RGBColor.from_string('26313B')
        if p.style.name.startswith('Heading'):
            level=int(p.style.name.split()[-1])
            fmt.keep_with_next=True
            fmt.space_before=Pt(12 if level==1 else 8)
            fmt.space_after=Pt(5)
            for run in p.runs:
                run.font.size=Pt({1:17,2:12.5,3:10.5}.get(level,10))
                run.bold=True
                run.font.color.rgb=RGBColor.from_string('234C59')
            if level==1:
                borders=OxmlElement('w:pBdr')
                line=OxmlElement('w:bottom')
                for k,v in [('val','single'),('sz','6'),('space','5'),('color','B9CED3')]:
                    line.set(qn('w:'+k),v)
                borders.append(line)
                p._p.get_or_add_pPr().append(borders)
        elif p._p.xpath('.//w:drawing'):
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            fmt.space_before=Pt(4)
            fmt.space_after=Pt(2)
            fmt.keep_with_next=True
        elif p.text.startswith(('图 ', '样例直接读取')):
            fmt.space_after=Pt(7)
            for run in p.runs:
                run.font.size=Pt(8)
                run.font.color.rgb=RGBColor.from_string('66747B')
    # A compact title block instead of a mostly empty cover page.
    for p in doc.paragraphs[:2]:
        p.paragraph_format.space_after=Pt(8)
        for run in p.runs:
            run.font.size=Pt(18 if p==doc.paragraphs[0] else 11)
            run.bold=True
    for table in doc.tables:
        table.autofit=False
        count=len(table.columns)
        widths=([2.6]+[(17.6-2.6)/(count-1)]*(count-1)) if count>=5 else [17.6/count]*count
        for column,width in zip(table.columns,widths):
            column.width=Cm(width)
        for i,row in enumerate(table.rows):
            for cell,width in zip(row.cells,widths):
                cell.width=Cm(width)
                props=cell._tc.get_or_add_tcPr()
                for old in list(props.findall(qn('w:shd'))):
                    props.remove(old)
                shade=OxmlElement('w:shd')
                shade.set(qn('w:fill'),'DFEBEE' if i==0 else ('F3F6F7' if i%2==0 else 'FFFFFF'))
                props.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next=(i==0)
                    p.paragraph_format.space_before=Pt(1)
                    p.paragraph_format.space_after=Pt(1)
                    p.paragraph_format.line_spacing=1.05
                    for run in p.runs:
                        run.font.size=Pt(7.5 if count>=6 else 8.5)
                        run.font.name='Calibri'
                        run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
                        run.font.color.rgb=RGBColor.from_string('26313B')
    for shape in doc.inline_shapes:
        # Keep plots legible but avoid consuming a whole page per figure.
        factor=min(1, Cm(16.5)/shape.width, Cm(11)/shape.height)
        shape.width=int(shape.width*factor)
        shape.height=int(shape.height*factor)
    output=source.with_name(source.name.replace('-特征数值说明补充版-排版优化版','-紧凑设计版'))
    doc.save(output)
    check=Document(output)
    assert content(check)==before
    assert len(check.inline_shapes)==images
    assert not any(p._p.xpath('.//w:br[@w:type="page"]') for p in check.paragraphs)
    print(output.name, 'content preserved; manual page breaks removed;',len(check.tables),'tables;',images,'images')
