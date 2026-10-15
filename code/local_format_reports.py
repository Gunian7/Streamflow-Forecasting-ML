from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT = Path(__file__).resolve().parent.parent

def set_font(run, size, bold=None):
    run.font.name = 'Times New Roman'
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold

def texts(doc):
    return [p.text for p in doc.paragraphs], [[c.text for r in t.rows for c in r.cells] for t in doc.tables]

for path in ROOT.glob('*特征数值说明补充版.docx'):
    doc = Document(path)
    before = texts(doc)
    images = len(doc.inline_shapes)
    for sec in doc.sections:
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.2)
        sec.left_margin, sec.right_margin = Cm(2), Cm(2)
        sec.header_distance, sec.footer_distance = Cm(1), Cm(1)
    for p in doc.paragraphs:
        fmt = p.paragraph_format
        fmt.widow_control = True
        if p.style.name.startswith('Heading'):
            level = int(p.style.name.split()[-1])
            fmt.keep_with_next = True
            fmt.space_before, fmt.space_after = Pt(14 if level == 1 else 10), Pt(6)
            fmt.line_spacing = 1.15
            for run in p.runs:
                set_font(run, {1:16,2:13,3:11}.get(level,11), True)
                run.font.color.rgb = RGBColor(0,0,0)
        elif p._p.xpath('.//w:drawing'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt.space_before, fmt.space_after = Pt(6), Pt(4)
            fmt.keep_with_next = True
        elif p.text.startswith('图 '):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt.space_after = Pt(10)
            for run in p.runs:
                set_font(run,9)
        elif p.alignment != WD_ALIGN_PARAGRAPH.CENTER:
            fmt.line_spacing = 1.35
            fmt.space_after = Pt(6)
            for run in p.runs:
                set_font(run,10.5)
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        count = len(table.columns)
        widths = ([2.5]+[(17-2.5)/(count-1)]*(count-1)) if count >= 5 else [17/count]*count
        for column, width in zip(table.columns,widths):
            column.width = Cm(width)
        header = table.rows[0]._tr.get_or_add_trPr()
        repeat = OxmlElement('w:tblHeader')
        header.append(repeat)
        for index,row in enumerate(table.rows):
            props = row._tr.get_or_add_trPr()
            props.append(OxmlElement('w:cantSplit'))
            for cell,width in zip(row.cells,widths):
                cell.width = Cm(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr = cell._tc.get_or_add_tcPr()
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), 'EAEFF4' if index == 0 else ('F7F8FA' if index % 2 == 0 else 'FFFFFF'))
                tcpr.append(shade)
                margins = OxmlElement('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    node=OxmlElement('w:'+edge)
                    node.set(qn('w:w'),'65')
                    node.set(qn('w:type'),'dxa')
                    margins.append(node)
                tcpr.append(margins)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.line_spacing = 1.1
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if index == 0 or count >= 5 else WD_ALIGN_PARAGRAPH.LEFT
                    for run in p.runs:
                        set_font(run,8 if count >= 6 else 9,index == 0)
        # Keep column labels attached to the first data row.
        for cell in table.rows[0].cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next=True
    for shape in doc.inline_shapes:
        if shape.width > Cm(17):
            ratio = Cm(17)/shape.width
            shape.height = int(shape.height*ratio)
            shape.width = Cm(17)
    footer = doc.sections[0].footer.paragraphs[0]
    if not footer.text and not footer._p.xpath('.//w:fldChar'):
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer.add_run('第 ')
        run=footer.add_run()
        field=OxmlElement('w:fldSimple')
        field.set(qn('w:instr'),'PAGE')
        run._r.append(field)
        footer.add_run(' 页')
    out=path.with_stem(path.stem+'-排版优化版')
    doc.save(out)
    check=Document(out)
    assert texts(check)==before, '正文或表格文本改变'
    assert len(check.inline_shapes)==images
    print(out.name, '文本与数值保持一致', len(check.tables), '表', images, '图')
