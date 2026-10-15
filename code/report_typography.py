from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH


def formal_typography(doc):
    def font(run, chinese, size, bold=False):
        run.font.name='Times New Roman'
        run.font.size=Pt(size)
        run.bold=bold
        run.font.color.rgb=RGBColor(0,0,0)
        fonts=run._element.get_or_add_rPr().rFonts
        for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
            fonts.attrib.pop(qn('w:'+attr),None)
        for attr,value in [('ascii','Times New Roman'),('hAnsi','Times New Roman'),('eastAsia',chinese)]:
            fonts.set(qn('w:'+attr),value)
    for name in ['Normal','Title','Heading 1','Heading 2','Heading 3']:
        style=doc.styles[name]
        style.font.name='Times New Roman'
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体' if name=='Normal' else '黑体')
    for p in doc.paragraphs:
        title=p.style.name=='Title'
        heading=p.style.name.startswith('Heading')
        caption=p.text.startswith(('ANN 两种','RF 两种','ANN 直接策略第','ANN 多输出策略第','RF 直接策略第','RF 多输出策略第','候选编号对应'))
        p.paragraph_format.line_spacing=1.3
        p.paragraph_format.space_after=Pt(6)
        p.paragraph_format.widow_control=True
        if heading or title:
            p.paragraph_format.keep_with_next=True
            p.paragraph_format.space_before=Pt(12)
            size=20 if title else (15 if p.style.name=='Heading 1' else 12)
            for r in p.runs: font(r,'黑体',size,True)
        else:
            p.paragraph_format.first_line_indent=Pt(21) if p.text and not caption and not p._p.xpath('.//w:drawing') else Pt(0)
            if p._p.xpath('.//w:drawing') or caption: p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs: font(r,'宋体',9 if caption else 10.5)
    for table in doc.tables:
        for i,row in enumerate(table.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent=Pt(0)
                    p.paragraph_format.line_spacing=1.1
                    p.paragraph_format.space_after=Pt(3)
                    for r in p.runs: font(r,'黑体' if i==0 else '宋体',9,i==0)
