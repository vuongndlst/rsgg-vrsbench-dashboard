import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from pathlib import Path

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def polish_document(doc_path: Path):
    doc = docx.Document(doc_path)

    # Set page margins to standard 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Polish paragraphs and headings
    for p in doc.paragraphs:
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        
        # Check headings
        if p.style.name.startswith('Heading 1'):
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(14)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79) # Deep Navy
        elif p.style.name.startswith('Heading 2'):
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(12)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        elif p.style.name.startswith('Heading 3'):
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(11)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x2A, 0x5C, 0x8A)
        elif p.style.name == 'Title':
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(12)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(18)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        else:
            for r in p.runs:
                r.font.name = 'Times New Roman'
                if not r.font.size:
                    r.font.size = Pt(11)

    # Polish tables
    for tbl in doc.tables:
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl, color="94A3B8", sz="6", val="single")

        # Format header row
        if len(tbl.rows) > 0:
            header_row = tbl.rows[0]
            header_tr = header_row._tr.get_or_add_trPr()
            header_tr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            for cell in header_row.cells:
                set_cell_background(cell, "0F172A") # Dark Navy / Slate
                set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
                for cp in cell.paragraphs:
                    cp.paragraph_format.space_after = Pt(0)
                    for r in cp.runs:
                        r.font.name = 'Times New Roman'
                        r.font.bold = True
                        r.font.size = Pt(10)
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) # White text

        # Format body rows
        for r_idx, row in enumerate(tbl.rows[1:]):
            is_ours = any("RSGG" in cell.text or "Ours" in cell.text for cell in row.cells)
            bg_color = "E0F2FE" if is_ours else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
            
            for cell in row.cells:
                if is_ours:
                    set_cell_background(cell, "F0FDF4") # Subtle emerald/mint for SOTA row
                elif r_idx % 2 == 1:
                    set_cell_background(cell, "F8FAFC")
                set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
                for cp in cell.paragraphs:
                    cp.paragraph_format.space_after = Pt(0)
                    for r in cp.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(9.5)
                        if is_ours:
                            r.font.bold = True
                            if "★" in r.text or "%" in r.text or "." in r.text:
                                r.font.color.rgb = RGBColor(0x04, 0x78, 0x57) # Emerald highlight

    doc.save(doc_path)
    print(f"Successfully polished and saved IEEE report: {doc_path} ({doc_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    p = Path("reports/BAO_CAO_KHOA_HOC_RSGG_IEEE_VIETNAMESE.docx")
    polish_document(p)

