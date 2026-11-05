# -*- coding: utf-8 -*-
"""Word 排版基座：《机器学习课程实践》三次阶段验收文档共用样式与构件。

只暴露函数（无导入副作用）：导入本模块不会创建文件、不会注册全局状态。

排版基线
    * 正文：中文宋体（SimSun）小四 12pt，行距 1.5 倍，首行缩进 2 字符
    * 西文/数字：Times New Roman；标题与正文统一宋体加粗体系
    * 表格：三线表（顶线粗 1.5pt、表头下横线细 0.75pt、底线粗 1.5pt），表题在表上方居中
    * 图片：图题在图下方居中；图注再小一号（小五 9pt）
    * 页面：A4，页边距上下 2.54cm、左右 3.17cm，页码居中页脚
"""

import os
from pathlib import Path

from docx import Document
from docx.enum.section import WD_HEADER_FOOTER, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ---------------------------------------------------------------- 常量

EAST_ASIA_FONT = "宋体"           # 中文正文/标题统一宋体
LATIN_FONT = "Times New Roman"    # 西文与数字

BODY_SIZE = Pt(12)                # 小四
BODY_LINE_SPACING = 1.5
BODY_FIRST_LINE_CHARS = 2         # 首行缩进 2 字符
BODY_FIRST_LINE_PT = 24           # 2 字符 × 12pt 的等效磅值（回落用）

H1_SIZE = Pt(16)                  # 三号
H2_SIZE = Pt(14)                  # 四号

CAPTION_SIZE = Pt(10.5)           # 五号：图题/表题
NOTE_SIZE = Pt(9)                 # 小五：图注
TABLE_SIZE = Pt(10.5)             # 五号：表内文字

COVER_TITLE_SIZE = Pt(22)
COVER_SUBTITLE_SIZE = Pt(16)
COVER_FIELD_SIZE = Pt(14)

PAGE_MARGIN_TB = Cm(2.54)
PAGE_MARGIN_LR = Cm(3.17)
PAGE_WIDTH_CM = 21.0
PAGE_HEIGHT_CM = 29.7
TEXT_WIDTH_CM = PAGE_WIDTH_CM - 2 * 3.17   # 14.66cm

THICK_LINE_SZ = "12"   # 1.5 磅 = 12 个 1/8 磅
THIN_LINE_SZ = "6"     # 0.75 磅 = 6 个 1/8 磅

THEME_ATTRS = ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme")

# OOXML 子元素顺序（Word 对 tblPr/tcPr 内的顺序敏感，乱序可能被静默忽略）
_TBLPR_SEQ = ("w:tblStyle", "w:tblpPr", "w:tblOverlap", "w:bidiVisual",
              "w:tblStyleRowBandSize", "w:tblStyleColBandSize", "w:tblW", "w:jc",
              "w:tblCellSpacing", "w:tblInd", "w:tblBorders", "w:shd", "w:tblLayout",
              "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription")
_TCPR_SEQ = ("w:cnfStyle", "w:tcW", "w:gridSpan", "w:hMerge", "w:vMerge", "w:tcBorders",
             "w:shd", "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign")

# 节属性顺序：w:pgNumType 必须落在 w:pgMar 之后、w:cols 之前，否则 Word 会忽略
_SECTPR_SEQ = ("w:headerReference", "w:footerReference", "w:footnotePr", "w:endnotePr",
               "w:type", "w:pgSz", "w:pgMar", "w:paperSrc", "w:pgBorders", "w:lnNumType",
               "w:pgNumType", "w:cols", "w:formProt", "w:vAlign", "w:noEndnote",
               "w:titlePg", "w:textDirection", "w:bidi", "w:rtlGutter", "w:docGrid",
               "w:printerSettings", "w:sectPrChange")


# ---------------------------------------------------------------- 底层工具


_LATIN_CACHE = {}


def latin_font_name():
    """西文字体名：有 Times New Roman 用它，缺失时回退宋体（只查一次并缓存）。"""
    if "value" not in _LATIN_CACHE:
        candidates = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
                      Path("/usr/share/fonts")]
        present = any(d.is_dir() and any(d.glob("times*.ttf")) for d in candidates)
        _LATIN_CACHE["value"] = LATIN_FONT if present else EAST_ASIA_FONT
    return _LATIN_CACHE["value"]


def _insert_ordered(parent, element, seq):
    """按 OOXML 顺序把 element 插到 parent 中（先找到它之后的第一个兄弟）。"""
    tag = element.tag
    ordered = list(seq)
    index = next(i for i, t in enumerate(ordered) if qn(t) == tag)
    for successor in ordered[index + 1:]:
        found = parent.find(qn(successor))
        if found is not None:
            found.addprevious(element)
            return element
    parent.append(element)
    return element


def _set_rfonts(rPr, east_asia=EAST_ASIA_FONT, latin=None):
    """写 w:rFonts：中文走 eastAsia，西文走 ascii/hAnsi，并清掉主题字体属性。"""
    latin = latin or latin_font_name()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    for attr in THEME_ATTRS:
        rFonts.attrib.pop(qn("w:" + attr), None)
    rFonts.set(qn("w:ascii"), latin)
    rFonts.set(qn("w:hAnsi"), latin)
    rFonts.set(qn("w:eastAsia"), east_asia)
    rFonts.set(qn("w:cs"), latin)
    return rFonts


def _blacken(rPr):
    """标题样式默认带主题色，这里统一改纯黑并剥掉主题属性。"""
    color = rPr.find(qn("w:color"))
    if color is not None:
        for attr in ("themeColor", "themeShade", "themeTint"):
            color.attrib.pop(qn("w:" + attr), None)


def set_run_font(run, size, bold=False, east_asia=EAST_ASIA_FONT, latin=None):
    """设定单个 run 的中西文字体、字号、加粗与纯黑颜色。

    中文必须写在 w:rFonts/@w:eastAsia 上：python-docx 的 run.font.name 只覆盖
    ascii/hAnsi，不设 eastAsia 会让中文回落为主题字体（等线/Calibri）。
    """
    run.font.size = size
    run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    rPr = run._element.get_or_add_rPr()
    _set_rfonts(rPr, east_asia, latin)
    _blacken(rPr)
    return run


def _style_font(style, size, bold=False, east_asia=EAST_ASIA_FONT, latin=None):
    """配置段落样式（Normal / Heading N）的中西文字体、字号、加粗。"""
    style.font.size = size
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rPr = style.element.get_or_add_rPr()
    _set_rfonts(rPr, east_asia, latin)
    _blacken(rPr)
    return style


def _set_first_line_chars(paragraph, chars):
    """补写 w:ind/@w:firstLineChars，让缩进真正按“字符”生效（Word 优先于 firstLine）。"""
    pPr = paragraph._p.get_or_add_pPr()
    ind = pPr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        pPr.append(ind)
    ind.set(qn("w:firstLineChars"), str(int(chars * 100)))
    return ind


def _add_page_field(paragraph):
    """在段落内插入 PAGE 域（begin/instrText/separate/占位文本/end）。"""
    run = paragraph.add_run()
    set_run_font(run, Pt(10.5), bold=False)

    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE   \\* MERGEFORMAT "
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")

    for node in (begin, instr, sep, placeholder, end):
        run._r.append(node)
    return run


# ---------------------------------------------------------------- 编号状态


class Numbering:
    """按章连续的图/表编号计数器（图 1-1、表 2-3）。显式实例化，模块不留全局状态。"""

    def __init__(self, chapter=1):
        self.chapter = chapter
        self._figure_seq = {}
        self._table_seq = {}

    def set_chapter(self, chapter):
        self.chapter = chapter
        return self

    def next_figure(self, chapter=None):
        ch = self.chapter if chapter is None else chapter
        self._figure_seq[ch] = self._figure_seq.get(ch, 0) + 1
        return "图 %d-%d" % (ch, self._figure_seq[ch])

    def next_table(self, chapter=None):
        ch = self.chapter if chapter is None else chapter
        self._table_seq[ch] = self._table_seq.get(ch, 0) + 1
        return "表 %d-%d" % (ch, self._table_seq[ch])


# ---------------------------------------------------------------- 文档与页面


def new_document():
    """新建 A4 文档：页边距、正文样式、标题样式、页码居中页脚一并就位。"""
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(PAGE_WIDTH_CM)
    section.page_height = Cm(PAGE_HEIGHT_CM)
    section.top_margin = PAGE_MARGIN_TB
    section.bottom_margin = PAGE_MARGIN_TB
    section.left_margin = PAGE_MARGIN_LR
    section.right_margin = PAGE_MARGIN_LR
    section.header_distance = Cm(1.5)
    section.footer_distance = Cm(1.75)

    normal = doc.styles["Normal"]
    _style_font(normal, BODY_SIZE, bold=False)
    pf = normal.paragraph_format
    pf.line_spacing = BODY_LINE_SPACING
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for name, size in (("Heading 1", H1_SIZE), ("Heading 2", H2_SIZE)):
        style = doc.styles[name]
        _style_font(style, size, bold=True)
        spf = style.paragraph_format
        spf.line_spacing = 1.5
        spf.first_line_indent = Pt(0)
        spf.keep_with_next = True

    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.first_line_indent = Pt(0)
    fp.paragraph_format.line_spacing = 1.0
    _add_page_field(fp)

    return doc


def add_page_break(doc):
    """插入分页符，返回承载分页符的段落。"""
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.add_run().add_break(WD_BREAK.PAGE)
    return paragraph


def _clear_footer(footer):
    """清空页脚各段落的全部内容（run 与域），只保留空段落外壳。"""
    for paragraph in footer.paragraphs:
        for tag in ("w:r", "w:fldSimple", "w:hyperlink"):
            for child in paragraph._p.findall(qn(tag)):
                paragraph._p.remove(child)
    return footer


def _fill_footer_page_number(footer):
    """把页脚首段做成居中的 PAGE 页码域。"""
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = paragraph.paragraph_format
    pf.first_line_indent = Pt(0)
    pf.line_spacing = 1.0
    _add_page_field(paragraph)
    return paragraph


def _set_section_page_start(section, start):
    """写 w:pgNumType/@w:start，让本节页码从 start 重新起编。"""
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    pgNumType = _insert_ordered(sectPr, OxmlElement("w:pgNumType"), _SECTPR_SEQ)
    pgNumType.set(qn("w:start"), str(int(start)))
    return pgNumType


def begin_body_section(doc, restart_at=1):
    """在当前位置分节（新页）：封面节用空白页脚，正文节页码从 restart_at 重新起编。

    在封面内容之后、正文之前调用一次；不要再额外调用 add_page_break（分节符已换页）。
    返回正文所属的新 Section。

    实现要点：Document.add_section 会把原 sectPr 克隆进新段落，两节因此共用同一个
    footer part；而 is_linked_to_previous=True 走的是 drop_rel，会把前一节仍在引用的
    关系一并删掉（遗留悬空 rId）。所以这里直接给新节挂一个全新的空页脚 part，
    前一节的页脚只做内容清空，不动关系。
    """
    new_section = doc.add_section(WD_SECTION.NEW_PAGE)

    _footer_part, r_id = doc.part.add_footer_part()
    sectPr = new_section._sectPr
    reference = sectPr.get_footerReference(WD_HEADER_FOOTER.PRIMARY)
    if reference is not None:
        reference.rId = r_id          # 改指向新 part，旧 part 仍归封面节
    else:
        sectPr.add_footerReference(WD_HEADER_FOOTER.PRIMARY, r_id)

    if len(doc.sections) >= 2:
        _clear_footer(doc.sections[-2].footer)   # 封面节：空白页脚，无 PAGE 域

    _set_section_page_start(new_section, restart_at)
    _fill_footer_page_number(new_section.footer)
    return new_section


# ---------------------------------------------------------------- 封面


COVER_FIELDS = ("学院", "专业", "班级", "指导教师", "学号", "姓名", "实践项目", "报告内容")


def _cover_label(text):
    """两字字段用全角空格撑成“学　　院”，与四字字段视觉等宽。"""
    return text[0] + "\u3000\u3000" + text[1:] if len(text) == 2 else text


def _set_cell_bottom_border(cell, sz=THIN_LINE_SZ):
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(old)
    tcBorders = _insert_ordered(tcPr, OxmlElement("w:tcBorders"), _TCPR_SEQ)
    el = OxmlElement("w:bottom")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), sz)
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")
    tcBorders.append(el)
    return tcBorders


def _set_table_noborder(table):
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = _insert_ordered(tblPr, OxmlElement("w:tblBorders"), _TBLPR_SEQ)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "auto")
        borders.append(el)
    return borders


def _plain_para(paragraph, align=None, line_spacing=1.5, space_before=0, space_after=0):
    if align is not None:
        paragraph.alignment = align
    pf = paragraph.paragraph_format
    pf.first_line_indent = Pt(0)
    pf.line_spacing = line_spacing
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    return paragraph


def add_cover(doc, college="", major="", class_name="", advisor="", student_ids="",
              names="", project="", content="", title=None, subtitle=None, date_text=None):
    """生成封面：主标题 + 副标题 + 8 行字段（值带下划线）+ 日期。

    字段顺序固定：学院 / 专业 / 班级 / 指导教师 / 学号 / 姓名 / 实践项目 / 报告内容。
    """
    if title is None:
        title = "杭州电子科技大学《机器学习课程实践》阶段验收报告"

    p = doc.add_paragraph()
    _plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 60, 12)
    set_run_font(p.add_run(title), COVER_TITLE_SIZE, bold=True)

    if subtitle:
        p = doc.add_paragraph()
        _plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 0, 36)
        set_run_font(p.add_run(subtitle), COVER_SUBTITLE_SIZE, bold=True)

    values = {"学院": college, "专业": major, "班级": class_name, "指导教师": advisor,
              "学号": student_ids, "姓名": names, "实践项目": project, "报告内容": content}

    table = doc.add_table(rows=len(COVER_FIELDS), cols=2)
    table.style = doc.styles["Normal Table"]
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _set_table_noborder(table)

    label_width, value_width = Cm(4.2), Cm(8.4)
    for row, field in zip(table.rows, COVER_FIELDS):
        row.height = Cm(1.15)
        for idx, cell in enumerate(row.cells):
            cell.width = label_width if idx == 0 else value_width
            para = _plain_para(cell.paragraphs[0], None, 1.5, 0, 0)
            if idx == 0:
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                set_run_font(para.add_run(_cover_label(field)), COVER_FIELD_SIZE, bold=False)
            else:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_run_font(para.add_run(values.get(field, "")), COVER_FIELD_SIZE, bold=False)
                _set_cell_bottom_border(cell)

    if date_text:
        p = doc.add_paragraph()
        _plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 48, 0)
        set_run_font(p.add_run(date_text), COVER_FIELD_SIZE, bold=False)

    return doc


# ---------------------------------------------------------------- 标题与正文


def _heading(doc, text, level, size):
    paragraph = doc.add_paragraph(style="Heading %d" % level)
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.LEFT, 1.5, 12 if level == 1 else 9, 6)
    paragraph.paragraph_format.keep_with_next = True
    set_run_font(paragraph.add_run(text), size, bold=True)
    return paragraph


def add_heading1(doc, text):
    """一级标题：宋体加粗 三号 16pt。"""
    return _heading(doc, text, 1, H1_SIZE)


def add_heading2(doc, text):
    """二级标题：宋体加粗 四号 14pt。"""
    return _heading(doc, text, 2, H2_SIZE)


def add_body(doc, text, indent=True, size=BODY_SIZE, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """正文段落：宋体小四 12pt、1.5 倍行距、首行缩进 2 字符（indent=False 可关闭）。"""
    paragraph = doc.add_paragraph()
    _plain_para(paragraph, align, BODY_LINE_SPACING, 0, 0)
    if indent:
        paragraph.paragraph_format.first_line_indent = Pt(BODY_FIRST_LINE_PT)
        _set_first_line_chars(paragraph, BODY_FIRST_LINE_CHARS)
    set_run_font(paragraph.add_run(text), size, bold=False)
    return paragraph


# ---------------------------------------------------------------- 三线表


def add_table_caption(doc, caption, width_cm=TEXT_WIDTH_CM):
    """表题：表格上方、居中、与表格同宽（左右缩进归零）。"""
    paragraph = doc.add_paragraph()
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 6, 3)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.left_indent = Cm(0)
    paragraph.paragraph_format.right_indent = Cm(0)
    set_run_font(paragraph.add_run(caption), CAPTION_SIZE, bold=True)
    return paragraph


def _set_table_borders(table, header_rows=1):
    """三线表：表级只保留顶线/底线，表头行单元格补一条细横线，其余全部无框。"""
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = _insert_ordered(tblPr, OxmlElement("w:tblBorders"), _TBLPR_SEQ)
    for edge, sz in (("top", THICK_LINE_SZ), ("bottom", THICK_LINE_SZ)):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), sz)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    for edge in ("left", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "auto")
        borders.append(el)

    for row in list(table.rows)[:header_rows]:
        for cell in row.cells:
            _set_cell_bottom_border(cell, THIN_LINE_SZ)

    for old in tblPr.findall(qn("w:tblCellMar")):
        tblPr.remove(old)
    cellMar = _insert_ordered(tblPr, OxmlElement("w:tblCellMar"), _TBLPR_SEQ)
    for edge, width in (("top", "0"), ("left", "80"), ("bottom", "0"), ("right", "80")):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:w"), width)
        el.set(qn("w:type"), "dxa")
        cellMar.append(el)
    return borders


def _set_table_width(table, total_cm, col_count):
    """固定布局、全表宽、等宽列。"""
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblLayout")):
        tblPr.remove(old)
    layout = _insert_ordered(tblPr, OxmlElement("w:tblLayout"), _TBLPR_SEQ)
    layout.set(qn("w:type"), "fixed")

    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tblW = _insert_ordered(tblPr, OxmlElement("w:tblW"), _TBLPR_SEQ)
    tblW.set(qn("w:w"), str(int(total_cm * 567)))   # 1cm = 567 twips
    tblW.set(qn("w:type"), "dxa")

    col_cm = total_cm / col_count
    for column in table.columns:
        column.width = Cm(col_cm)
    for row in table.rows:
        for cell in row.cells:
            cell.width = Cm(col_cm)
    return table


def _fill_cell(cell, text, bold=False):
    para = _plain_para(cell.paragraphs[0], WD_ALIGN_PARAGRAPH.CENTER, 1.0, 1, 1)
    set_run_font(para.add_run(str(text)), TABLE_SIZE, bold=bold)
    return para


def add_table(doc, header, rows, caption=None, chapter=1, numbering=None,
              width_cm=TEXT_WIDTH_CM, header_rows=1):
    """生成三线表；给出 caption 时自动编号（“表 1-1  标题”）并置于表格上方。

    header: 表头文本序列；rows: 二维文本序列。
    numbering 为 None 时 caption 原样输出（由调用方自带“表 x-y”前缀）。
    返回 (table, caption_text)。
    """
    cap_text = None
    if caption is not None:
        if numbering is not None:
            cap_text = "%s  %s" % (numbering.next_table(chapter), caption)
        else:                     # 未给编号器时按调用方原文输出（调用方自带编号）
            cap_text = caption
        add_table_caption(doc, cap_text, width_cm)

    table = doc.add_table(rows=len(rows) + header_rows, cols=len(header))
    table.style = doc.styles["Normal Table"]
    _set_table_width(table, width_cm, len(header))
    _set_table_borders(table, header_rows)

    for col, text in enumerate(header):
        _fill_cell(table.cell(0, col), text, bold=True)
    for r, row_values in enumerate(rows, start=header_rows):
        for col, text in enumerate(row_values):
            _fill_cell(table.cell(r, col), text, bold=False)

    return table, cap_text


# ---------------------------------------------------------------- 插图


def add_figure(doc, image_path, caption=None, chapter=1, numbering=None,
               width_cm=TEXT_WIDTH_CM - 0.6):
    """插入插图；给出 caption 时自动编号并置于图下方居中。

    image_path 可为路径，也可为已打开的文件对象（如内存中的 BytesIO）。
    返回 (image_paragraph, caption_text)。
    """
    source = image_path if hasattr(image_path, "read") else str(image_path)

    paragraph = doc.add_paragraph()
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 6, 3)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(source, width=Cm(width_cm))

    cap_text = None
    if caption is not None:
        if numbering is not None:
            cap_text = "%s  %s" % (numbering.next_figure(chapter), caption)
        else:                     # 未给编号器时按调用方原文输出（调用方自带编号）
            cap_text = caption
        cap_para = doc.add_paragraph()
        _plain_para(cap_para, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 3, 6)
        set_run_font(cap_para.add_run(cap_text), CAPTION_SIZE, bold=True)

    return paragraph, cap_text


def add_figure_note(doc, text):
    """图注：图题之下的补充说明，小五 9pt，居中，无首行缩进。"""
    paragraph = doc.add_paragraph()
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 0, 9)
    set_run_font(paragraph.add_run(text), NOTE_SIZE, bold=False)
    return paragraph


__all__ = [
    "new_document", "add_cover", "add_heading1", "add_heading2", "add_body",
    "add_table", "add_table_caption", "add_figure", "add_figure_note",
    "add_page_break", "begin_body_section", "set_run_font", "latin_font_name", "Numbering",
    "EAST_ASIA_FONT", "LATIN_FONT", "BODY_SIZE", "BODY_LINE_SPACING",
    "COVER_FIELDS",
]
