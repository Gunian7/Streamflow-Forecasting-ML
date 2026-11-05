# -*- coding: utf-8 -*-
"""Word 排版基座：《机器学习课程实践》三次阶段验收文档共用样式与构件。

只暴露函数（无导入副作用）：导入本模块不会创建文件、不会注册全局状态。

排版体系（正式论文口径）
    * 页面：A4 210×297mm，页边距上下 2.54cm、左右 3.17cm，页眉距边界 1.5cm、页脚 1.75cm
    * 页眉：宋体 9pt 居中，文字由调用方传入，段落带 0.5pt 单线下框线；封面节无页眉
    * 页脚：宋体 9pt 居中 PAGE 页码域，正文节从 1 起编
    * 一级标题：黑体 16pt 不加粗 居中，段前 18pt / 段后 12pt，与下段同页
    * 二级标题：黑体 14pt 不加粗 左对齐，段前 12pt / 段后 6pt，与下段同页
    * 三级标题：黑体 12pt 不加粗 左对齐，段前 6pt / 段后 3pt
    * 正文：宋体 12pt，1.5 倍行距，首行缩进 2 字符，两端对齐，段前段后 0
    * 表题：宋体 10.5pt 加粗 居中，段前 6pt / 段后 3pt，与表格同页
    * 表格：三线表（顶/底线 1.5pt，表头下 0.75pt，其余无线），表内宋体 10.5pt，
      表头加粗居中，全单元格垂直居中，整表居中，单元格左右边距 0.08cm，表头行跨页重复
    * 图片：内联居中，段前 6pt / 段后 3pt，宽度上限 14.6cm（超出按比例缩小）
    * 图题：宋体 10.5pt 加粗 居中，段前 3pt / 段后 3pt
    * 图注：宋体 9pt 左对齐，无首行缩进，段后 6pt
"""

import os
from pathlib import Path

from docx import Document
from docx.enum.section import WD_HEADER_FOOTER, WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Length, Pt, RGBColor

# ---------------------------------------------------------------- 常量

SONG_FONT = "宋体"                # 正文、表格、图表题、页眉页脚
HEI_FONT = "黑体"                 # 各级标题（黑体字面本身即粗体，故不再加粗）
EAST_ASIA_FONT = SONG_FONT        # 兼容旧名：默认中文字体
LATIN_FONT = "Times New Roman"    # 西文与数字

BODY_SIZE = Pt(12)                # 小四
BODY_LINE_SPACING = 1.5
BODY_FIRST_LINE_CHARS = 2         # 首行缩进 2 字符
BODY_FIRST_LINE_PT = 24           # 2 字符 × 12pt 的等效磅值（回落用）

H1_SIZE = Pt(16)                  # 三号
H1_SPACE_BEFORE, H1_SPACE_AFTER = Pt(18), Pt(12)
H2_SIZE = Pt(14)                  # 四号
H2_SPACE_BEFORE, H2_SPACE_AFTER = Pt(12), Pt(6)
H3_SIZE = Pt(12)                  # 小四
H3_SPACE_BEFORE, H3_SPACE_AFTER = Pt(6), Pt(3)

CAPTION_SIZE = Pt(10.5)           # 五号：图题/表题
CAPTION_SPACE_BEFORE, CAPTION_SPACE_AFTER = Pt(6), Pt(3)
FIGURE_CAPTION_SPACE_BEFORE, FIGURE_CAPTION_SPACE_AFTER = Pt(3), Pt(3)
NOTE_SIZE = Pt(9)                 # 小五：图注
NOTE_SPACE_BEFORE, NOTE_SPACE_AFTER = Pt(3), Pt(6)
TABLE_SIZE = Pt(10.5)             # 五号：表内文字

HEADER_SIZE = Pt(9)
FOOTER_SIZE = Pt(9)

COVER_TITLE_SIZE = Pt(22)
COVER_SUBTITLE_SIZE = Pt(16)
COVER_FIELD_SIZE = Pt(12)
COVER_DATE_SIZE = Pt(14)

PAGE_MARGIN_TB = Cm(2.54)
PAGE_MARGIN_LR = Cm(3.17)
PAGE_HEADER_DISTANCE = Cm(1.5)
PAGE_FOOTER_DISTANCE = Cm(1.75)
PAGE_WIDTH_CM = 21.0
PAGE_HEIGHT_CM = 29.7
TEXT_WIDTH_CM = PAGE_WIDTH_CM - 2 * 3.17   # 14.66cm
FIGURE_MAX_WIDTH_CM = 14.6                 # 图片宽度上限
TABLE_CELL_MARGIN_CM = 0.08                # 单元格左右边距

THICK_LINE_SZ = "12"   # 1.5 磅 = 12 个 1/8 磅
THIN_LINE_SZ = "6"     # 0.75 磅 = 6 个 1/8 磅
HEADER_LINE_SZ = "4"   # 0.5 磅 = 4 个 1/8 磅

TABLE_CELL_MARGIN_TWIPS = str(int(round(TABLE_CELL_MARGIN_CM * 567)))  # 1cm = 567 twips

THEME_ATTRS = ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme")

# OOXML 子元素顺序（Word 对 pPr/tblPr/tcPr 内的顺序敏感，乱序可能被静默忽略）
_PPR_SEQ = ("w:pStyle", "w:keepNext", "w:keepLines", "w:pageBreakBefore", "w:framePr",
            "w:widowControl", "w:numPr", "w:suppressLineNumbers", "w:pBdr", "w:shd",
            "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
            "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
            "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
            "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
            "w:textDirection", "w:textAlignment", "w:textboxTightWrap", "w:outlineLvl",
            "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange")
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

# 页眉文字待写槽：new_document(header_text=...) 先登记，begin_body_section 落到正文节
_PENDING_HEADER = {}


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


def _set_rfonts(rPr, east_asia=SONG_FONT, latin=None):
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


def set_run_font(run, size, bold=False, east_asia=SONG_FONT, latin=None):
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


def _style_font(style, size, bold=False, east_asia=SONG_FONT, latin=None):
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
        ind = _insert_ordered(pPr, OxmlElement("w:ind"), _PPR_SEQ)
    ind.set(qn("w:firstLineChars"), str(int(chars * 100)))
    return ind


def _clear_paragraph(paragraph):
    """清空段落的 run、域与超链接，只保留段落外壳（used by 页眉/页脚）。"""
    for tag in ("w:r", "w:fldSimple", "w:hyperlink"):
        for child in paragraph._p.findall(qn(tag)):
            paragraph._p.remove(child)
    return paragraph


def _set_paragraph_bottom_border(paragraph, sz=HEADER_LINE_SZ):
    """给段落加一条下框线（页眉下边线）。"""
    pPr = paragraph._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:pBdr")):
        pPr.remove(old)
    pBdr = _insert_ordered(pPr, OxmlElement("w:pBdr"), _PPR_SEQ)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), sz)
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "000000")
    pBdr.append(bottom)
    return pBdr


def _add_page_field(paragraph, size=FOOTER_SIZE):
    """在段落内插入 PAGE 域（begin/instrText/separate/占位文本/end）。"""
    run = paragraph.add_run()
    set_run_font(run, size, bold=False)

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


def new_document(header_text=None):
    """新建 A4 文档：页面、正文样式、标题样式、页脚页码一并就位。

    header_text 给出时登记为该文档的正文页眉文字，在 begin_body_section 时
    落到正文节（封面节始终无页眉）。也可稍后调用 set_header(doc, text)。
    """
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(PAGE_WIDTH_CM)
    section.page_height = Cm(PAGE_HEIGHT_CM)
    section.top_margin = PAGE_MARGIN_TB
    section.bottom_margin = PAGE_MARGIN_TB
    section.left_margin = PAGE_MARGIN_LR
    section.right_margin = PAGE_MARGIN_LR
    section.header_distance = PAGE_HEADER_DISTANCE
    section.footer_distance = PAGE_FOOTER_DISTANCE

    normal = doc.styles["Normal"]
    _style_font(normal, BODY_SIZE, bold=False, east_asia=SONG_FONT)
    pf = normal.paragraph_format
    pf.line_spacing = BODY_LINE_SPACING
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for name, size, before, after, align in (
            ("Heading 1", H1_SIZE, H1_SPACE_BEFORE, H1_SPACE_AFTER, WD_ALIGN_PARAGRAPH.CENTER),
            ("Heading 2", H2_SIZE, H2_SPACE_BEFORE, H2_SPACE_AFTER, WD_ALIGN_PARAGRAPH.LEFT),
            ("Heading 3", H3_SIZE, H3_SPACE_BEFORE, H3_SPACE_AFTER, WD_ALIGN_PARAGRAPH.LEFT)):
        style = doc.styles[name]
        _style_font(style, size, bold=False, east_asia=HEI_FONT)
        spf = style.paragraph_format
        spf.line_spacing = 1.5
        spf.first_line_indent = Pt(0)
        spf.space_before = before
        spf.space_after = after
        spf.alignment = align
        spf.keep_with_next = True

    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    _plain_para(fp, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 0, 0)
    _add_page_field(fp)

    if header_text:
        _PENDING_HEADER["text"] = header_text

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
        _clear_paragraph(paragraph)
    return footer


def _fill_footer_page_number(footer):
    """把页脚首段做成居中的 PAGE 页码域（宋体 9pt）。"""
    paragraph = footer.paragraphs[0]
    _clear_paragraph(paragraph)
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 0, 0)
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


def _fill_header(header, text):
    """把页眉首段做成宋体 9pt 居中 + 0.5pt 下框线的单行文字。"""
    paragraph = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    _clear_paragraph(paragraph)
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 0, 0)
    set_run_font(paragraph.add_run(text), HEADER_SIZE, bold=False, east_asia=SONG_FONT)
    _set_paragraph_bottom_border(paragraph)
    return paragraph


def _attach_header(doc, section, text):
    """给指定节挂一个独立的页眉 part 并写入文字（不影响其它节）。"""
    _header_part, r_id = doc.part.add_header_part()
    sectPr = section._sectPr
    reference = sectPr.get_headerReference(WD_HEADER_FOOTER.PRIMARY)
    if reference is not None:
        reference.rId = r_id
    else:
        sectPr.add_headerReference(WD_HEADER_FOOTER.PRIMARY, r_id)
    return _fill_header(section.header, text)


def begin_body_section(doc, restart_at=1, header_text=None):
    """在当前位置分节（新页）：封面节无页眉无页脚，正文节页码从 restart_at 起编。

    在封面内容之后、正文之前调用一次；不要再额外调用 add_page_break（分节符已换页）。
    返回正文所属的新 Section。

    实现要点：Document.add_section 会把原 sectPr 克隆进新段落，两节因此共用同一个
    header/footer part；而 is_linked_to_previous=True 走的是 drop_rel，会把前一节仍在
    引用的关系一并删掉（遗留悬空 rId）。所以这里直接给新节各挂一个全新的空 part，
    前一节的页眉页脚只做内容清空，不动关系。
    """
    new_section = doc.add_section(WD_SECTION.NEW_PAGE)

    _footer_part, footer_id = doc.part.add_footer_part()
    sectPr = new_section._sectPr
    reference = sectPr.get_footerReference(WD_HEADER_FOOTER.PRIMARY)
    if reference is not None:
        reference.rId = footer_id          # 改指向新 part，旧 part 仍归封面节
    else:
        sectPr.add_footerReference(WD_HEADER_FOOTER.PRIMARY, footer_id)

    if len(doc.sections) >= 2:
        cover = doc.sections[-2]
        _clear_footer(cover.footer)               # 封面节：空白页脚，无 PAGE 域
        for ref in list(cover._sectPr.findall(qn("w:headerReference"))):
            cover._sectPr.remove(ref)             # 封面节：不留页眉引用

    _set_section_page_start(new_section, restart_at)
    _fill_footer_page_number(new_section.footer)

    text = header_text if header_text is not None else _PENDING_HEADER.get("text")
    if text:
        _attach_header(doc, new_section, text)
    _PENDING_HEADER.pop("text", None)
    return new_section


def set_header(doc, text):
    """设置正文页眉文字（宋体 9pt 居中 + 下框线）。

    正文节已存在时立即写入该节；否则登记待写槽，由 begin_body_section 落地。
    封面节始终不显示页眉。
    """
    if not text:
        return None
    if len(doc.sections) >= 2:
        _PENDING_HEADER.pop("text", None)
        return _attach_header(doc, doc.sections[-1], text)
    _PENDING_HEADER["text"] = text
    return None


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
    pf.space_before = space_before if isinstance(space_before, Length) else Pt(space_before)
    pf.space_after = space_after if isinstance(space_after, Length) else Pt(space_after)
    return paragraph


def add_cover(doc, college="", major="", class_name="", advisor="", student_ids="",
              names="", project="", content="", title=None, subtitle=None, date_text=None):
    """生成封面：主标题 + 副标题 + 8 行字段（值带下划线）+ 日期。

    字段顺序固定：学院 / 专业 / 班级 / 指导教师 / 学号 / 姓名 / 实践项目 / 报告内容。
    标题与日期用黑体/宋体按论文口径排布，字段名与字段值均为宋体 12pt。
    """
    if title is None:
        title = "杭州电子科技大学《机器学习课程实践》阶段验收报告"

    p = doc.add_paragraph()
    _plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 60, 12)
    set_run_font(p.add_run(title), COVER_TITLE_SIZE, bold=False, east_asia=HEI_FONT)

    if subtitle:
        p = doc.add_paragraph()
        _plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 0, 36)
        set_run_font(p.add_run(subtitle), COVER_SUBTITLE_SIZE, bold=False, east_asia=HEI_FONT)

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
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = _plain_para(cell.paragraphs[0], None, 1.5, 0, 0)
            if idx == 0:
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                set_run_font(para.add_run(_cover_label(field)), COVER_FIELD_SIZE,
                             bold=False, east_asia=SONG_FONT)
            else:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_run_font(para.add_run(values.get(field, "")), COVER_FIELD_SIZE,
                             bold=False, east_asia=SONG_FONT)
                _set_cell_bottom_border(cell)

    if date_text:
        p = doc.add_paragraph()
        _plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 48, 0)
        set_run_font(p.add_run(date_text), COVER_DATE_SIZE, bold=False, east_asia=SONG_FONT)

    return doc


# ---------------------------------------------------------------- 标题与正文


_HEADING_SPECS = {
    1: (H1_SIZE, WD_ALIGN_PARAGRAPH.CENTER, H1_SPACE_BEFORE, H1_SPACE_AFTER),
    2: (H2_SIZE, WD_ALIGN_PARAGRAPH.LEFT, H2_SPACE_BEFORE, H2_SPACE_AFTER),
    3: (H3_SIZE, WD_ALIGN_PARAGRAPH.LEFT, H3_SPACE_BEFORE, H3_SPACE_AFTER),
}


def _heading(doc, text, level):
    size, align, before, after = _HEADING_SPECS[level]
    paragraph = doc.add_paragraph(style="Heading %d" % level)
    _plain_para(paragraph, align, BODY_LINE_SPACING, before, after)
    paragraph.paragraph_format.keep_with_next = True
    set_run_font(paragraph.add_run(text), size, bold=False, east_asia=HEI_FONT)
    return paragraph


def add_heading1(doc, text):
    """一级标题：黑体 16pt 不加粗 居中，段前 18pt、段后 12pt，与下段同页。"""
    return _heading(doc, text, 1)


def add_heading2(doc, text):
    """二级标题：黑体 14pt 不加粗 左对齐，段前 12pt、段后 6pt，与下段同页。"""
    return _heading(doc, text, 2)


def add_heading3(doc, text):
    """三级标题：黑体 12pt 不加粗 左对齐，段前 6pt、段后 3pt。"""
    return _heading(doc, text, 3)


def add_body(doc, text, indent=True, size=BODY_SIZE, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """正文段落：宋体小四 12pt、1.5 倍行距、首行缩进 2 字符（indent=False 可关闭）。"""
    paragraph = doc.add_paragraph()
    _plain_para(paragraph, align, BODY_LINE_SPACING, 0, 0)
    if indent:
        paragraph.paragraph_format.first_line_indent = Pt(BODY_FIRST_LINE_PT)
        _set_first_line_chars(paragraph, BODY_FIRST_LINE_CHARS)
    set_run_font(paragraph.add_run(text), size, bold=False, east_asia=SONG_FONT)
    return paragraph


# ---------------------------------------------------------------- 三线表


def add_table_caption(doc, caption, width_cm=TEXT_WIDTH_CM):
    """表题：表格上方、居中、宋体 10.5pt 加粗，段前 6pt、段后 3pt，与表格同页。"""
    paragraph = doc.add_paragraph()
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 1.0,
                CAPTION_SPACE_BEFORE.pt, CAPTION_SPACE_AFTER.pt)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.left_indent = Cm(0)
    paragraph.paragraph_format.right_indent = Cm(0)
    set_run_font(paragraph.add_run(caption), CAPTION_SIZE, bold=True, east_asia=SONG_FONT)
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
        _repeat_as_header_row(row)

    for old in tblPr.findall(qn("w:tblCellMar")):
        tblPr.remove(old)
    cellMar = _insert_ordered(tblPr, OxmlElement("w:tblCellMar"), _TBLPR_SEQ)
    for edge, width in (("top", "0"), ("left", TABLE_CELL_MARGIN_TWIPS),
                        ("bottom", "0"), ("right", TABLE_CELL_MARGIN_TWIPS)):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:w"), width)
        el.set(qn("w:type"), "dxa")
        cellMar.append(el)
    return borders


def _repeat_as_header_row(row):
    """把行标记为“跨页重复标题行”（w:trPr/w:tblHeader）。"""
    trPr = row._tr.get_or_add_trPr()
    for old in trPr.findall(qn("w:tblHeader")):
        trPr.remove(old)
    element = OxmlElement("w:tblHeader")
    element.set(qn("w:val"), "true")
    trPr.append(element)          # tblHeader 位于 trPr 尾部合法区间
    return element


def _column_widths_cm(rows_text, total_cm):
    """按列内容长度分配列宽（平方根阻尼），并保证每列不小于 1.8cm。"""
    n = len(rows_text[0])
    maxlen = []
    for j in range(n):
        longest = max((len(str(r[j])) for r in rows_text), default=2)
        maxlen.append(max(longest, 2))
    weights = [m ** 0.75 for m in maxlen]
    scale = total_cm / sum(weights)
    widths = [w * scale for w in weights]
    min_cm = 1.8
    # 先补齐过窄列，再从最宽列按比例扣减
    deficit = sum(min_cm - w for w in widths if w < min_cm)
    if deficit > 0:
        widths = [max(w, min_cm) for w in widths]
        surplus_idx = [i for i, w in enumerate(widths) if w > min_cm]
        surplus_total = sum(widths[i] - min_cm for i in surplus_idx)
        for i in surplus_idx:
            widths[i] -= deficit * (widths[i] - min_cm) / surplus_total
    scale = total_cm / sum(widths)
    return [w * scale for w in widths]


def _set_table_width(table, total_cm, col_count, rows_text=None):
    """固定布局、全表宽、整表居中；给出 rows_text 时按内容分配列宽。"""
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

    if rows_text:
        widths = _column_widths_cm(rows_text, total_cm)
    else:
        widths = [total_cm / col_count] * col_count
    for idx, column in enumerate(table.columns):
        column.width = Cm(widths[idx])
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            cell.width = Cm(widths[idx])
    return table


def _fill_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=None):
    """填一格；文本中的 "\n" 写成显式换行（w:br），用于强制逐条分行、避免长串被硬断。"""
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    para = _plain_para(cell.paragraphs[0], align, 1.0, 1, 1)
    font = size if size is not None else TABLE_SIZE
    for idx, piece in enumerate(str(text).split("\n")):
        if idx:
            set_run_font(para.add_run(), font, bold=bold, east_asia=SONG_FONT).add_break()
        set_run_font(para.add_run(piece), font, bold=bold, east_asia=SONG_FONT)
    return para


def add_table(doc, header, rows, caption=None, chapter=1, numbering=None,
              width_cm=TEXT_WIDTH_CM, header_rows=1, col_widths_cm=None, font_size=None):
    """生成三线表；给出 caption 时自动编号（“表 1-1  标题”）并置于表格上方。

    header: 表头文本序列；rows: 二维文本序列。
    numbering 为 None 时 caption 原样输出（由调用方自带“表 x-y”前缀）。
    col_widths_cm: 可选显式列宽（厘米，长度等于列数且合计等于 width_cm）；
        None 时沿用按内容长度阻尼分配的逻辑，既有调用方行为不变。
    font_size: 可选表内字号；None 时用 TABLE_SIZE（10.5pt）。
    返回 (table, caption_text)。
    """
    cap_text = None
    if caption is not None:
        if numbering is not None:
            cap_text = "%s  %s" % (numbering.next_table(chapter), caption)
        else:                     # 未给编号器时按调用方原文输出（调用方自带编号）
            cap_text = caption
        add_table_caption(doc, cap_text, width_cm)

    text_rows = [list(header)] + [list(r) for r in rows]
    table = doc.add_table(rows=len(rows) + header_rows, cols=len(header))
    table.style = doc.styles["Normal Table"]

    if col_widths_cm is None:
        _set_table_width(table, width_cm, len(header), text_rows)
        widths = _column_widths_cm(text_rows, width_cm)
    else:
        widths = [float(w) for w in col_widths_cm]
        if len(widths) != len(header):
            raise ValueError("col_widths_cm 长度 %d 与列数 %d 不一致" % (len(widths), len(header)))
        if abs(sum(widths) - width_cm) > 0.3:
            raise ValueError("col_widths_cm 合计 %.2fcm 与表宽 %.2fcm 不一致"
                             % (sum(widths), width_cm))
        _set_table_width(table, sum(widths), len(header), None)
        for idx, column in enumerate(table.columns):
            column.width = Cm(widths[idx])
        for row in table.rows:
            for idx, cell in enumerate(row.cells):
                cell.width = Cm(widths[idx])

    _set_table_borders(table, header_rows)

    for col, text in enumerate(header):
        _fill_cell(table.cell(0, col), text, bold=True, size=font_size)
    for r, row_values in enumerate(rows, start=header_rows):
        for col, text in enumerate(row_values):
            # 宽列（长文本，如“机理作用”“说明”）左对齐，窄列居中
            align = WD_ALIGN_PARAGRAPH.LEFT if widths[col] >= 4.2 else WD_ALIGN_PARAGRAPH.CENTER
            _fill_cell(table.cell(r, col), text, bold=False, align=align, size=font_size)

    return table, cap_text


# ---------------------------------------------------------------- 插图


def _place_picture(paragraph, source, cap_cm):
    """插入内联图片；宽度不超过 cap_cm，超出时按原比例缩小。"""
    picture = paragraph.add_run().add_picture(source)
    limit = Cm(cap_cm)
    if picture.width > limit:
        ratio = limit / picture.width
        picture.height = int(picture.height * ratio)
        picture.width = int(limit)
    return picture


def add_figure(doc, image_path, caption=None, chapter=1, numbering=None,
               width_cm=FIGURE_MAX_WIDTH_CM):
    """插入插图；给出 caption 时自动编号并置于图下方居中。

    image_path 可为路径，也可为已打开的文件对象（如内存中的 BytesIO）。
    图片宽度以 width_cm（默认上限 14.6cm）封顶，超出按比例缩小。
    返回 (image_paragraph, caption_text)。
    """
    source = image_path if hasattr(image_path, "read") else str(image_path)

    paragraph = doc.add_paragraph()
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 6, 3)
    paragraph.paragraph_format.keep_with_next = True
    _place_picture(paragraph, source, width_cm)

    cap_text = None
    if caption is not None:
        if numbering is not None:
            cap_text = "%s  %s" % (numbering.next_figure(chapter), caption)
        else:                     # 未给编号器时按调用方原文输出（调用方自带编号）
            cap_text = caption
        cap_para = doc.add_paragraph()
        _plain_para(cap_para, WD_ALIGN_PARAGRAPH.CENTER, 1.0,
                    FIGURE_CAPTION_SPACE_BEFORE.pt, FIGURE_CAPTION_SPACE_AFTER.pt)
        set_run_font(cap_para.add_run(cap_text), CAPTION_SIZE, bold=True, east_asia=SONG_FONT)

    return paragraph, cap_text


def add_figure_note(doc, text):
    """图注：图题之下的补充说明，宋体 9pt 左对齐，无首行缩进，段后 6pt。"""
    paragraph = doc.add_paragraph()
    _plain_para(paragraph, WD_ALIGN_PARAGRAPH.LEFT, 1.0,
                NOTE_SPACE_BEFORE.pt, NOTE_SPACE_AFTER.pt)
    set_run_font(paragraph.add_run(text), NOTE_SIZE, bold=False, east_asia=SONG_FONT)
    return paragraph


__all__ = [
    "new_document", "set_header", "add_cover", "add_heading1", "add_heading2",
    "add_heading3", "add_body", "add_table", "add_table_caption", "add_figure",
    "add_figure_note", "add_page_break", "begin_body_section", "set_run_font",
    "latin_font_name", "Numbering",
    "EAST_ASIA_FONT", "SONG_FONT", "HEI_FONT", "LATIN_FONT", "BODY_SIZE",
    "BODY_LINE_SPACING", "FIGURE_MAX_WIDTH_CM", "COVER_FIELDS",
]
