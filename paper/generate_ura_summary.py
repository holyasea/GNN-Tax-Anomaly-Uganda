import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_ura_summary_document(output_path):
    doc = docx.Document()

    # Page Margins: 1 inch (2.54 cm) throughout
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Color Palette: Executive Navy & Slate
    COLOR_NAVY = RGBColor(27, 54, 93)       # #1B365D - Formal URA/Academic Navy
    COLOR_SLATE = RGBColor(80, 95, 110)     # Subtitle Slate
    COLOR_BODY = RGBColor(45, 45, 45)       # Soft Charcoal for comfortable reading
    COLOR_LINK = RGBColor(0, 102, 204)      # Standard Hyperlink Blue

    # Set Default Normal Font
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = COLOR_BODY

    # Helper: Set paragraph spacing
    def format_spacing(p, before_pt=0, after_pt=6, line_mult=1.15):
        pPr = p._p.get_or_add_pPr()
        sp = OxmlElement('w:spacing')
        sp.set(qn('w:before'), str(int(before_pt * 20)))
        sp.set(qn('w:after'), str(int(after_pt * 20)))
        sp.set(qn('w:line'), str(int(line_mult * 240)))
        sp.set(qn('w:lineRule'), 'auto')
        pPr.append(sp)

    # -------------------------------------------------------------
    # 1. DOCUMENT HEADER & TITLE
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    format_spacing(p_title, before_pt=0, after_pt=6, line_mult=1.15)
    r_title = p_title.add_run("Graph Neural Networks for Anomaly Detection in Unlabeled Data: A Proof of Concept Study Using Uganda's Electronic Fiscal Receipting and Invoicing System Data")
    r_title.bold = True
    r_title.font.size = Pt(15)
    r_title.font.color.rgb = COLOR_NAVY

    p_sub = doc.add_paragraph()
    format_spacing(p_sub, before_pt=0, after_pt=12, line_mult=1.15)
    r_sub = p_sub.add_run("A Plain-Language Executive Summary for Uganda Revenue Authority (URA) Staff")
    r_sub.font.size = Pt(11.5)
    r_sub.italic = True
    r_sub.font.color.rgb = COLOR_SLATE

    # -------------------------------------------------------------
    # 2. METADATA CALLOUT BOX
    # -------------------------------------------------------------
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)

    cell = tbl.cell(0, 0)
    tcPr = cell._tc.get_or_add_tcPr()
    
    # Light gray-blue fill
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F9"/>')
    tcPr.append(shd)

    # Thick left navy border, no other borders
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)

    # Cell padding
    mar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="140" w:type="dxa"/>
            <w:left w:w="200" w:type="dxa"/>
            <w:bottom w:w="140" w:type="dxa"/>
            <w:right w:w="200" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(mar)

    p_meta = cell.paragraphs[0]
    format_spacing(p_meta, before_pt=2, after_pt=2, line_mult=1.15)
    
    run = p_meta.add_run("Author: ")
    run.bold = True
    p_meta.add_run("Holyweek Asea (Master of Statistics, Makerere University & Strategy and Risk Management Dept, URA)\n")
    
    run = p_meta.add_run("Official Email: ")
    run.bold = True
    p_meta.add_run("hasea@ura.go.ug | asea.holyweek@students.mak.ac.ug\n")

    run = p_meta.add_run("Supervisors: ")
    run.bold = True
    p_meta.add_run("Dr. Francis F. Bbosa, PhD and Dr. Douglas Candia Andabati, PhD\n")
    
    run = p_meta.add_run("Source Data: ")
    run.bold = True
    p_meta.add_run("Five years of administrative e-invoicing data from URA EFRIS (2018 to 2022)")

    # -------------------------------------------------------------
    # 3. EXECUTIVE SUMMARY (TWO PLAIN-LANGUAGE PARAGRAPHS)
    # -------------------------------------------------------------
    p_h1 = doc.add_paragraph()
    format_spacing(p_h1, before_pt=16, after_pt=6, line_mult=1.15)
    r_h1 = p_h1.add_run("Executive Summary")
    r_h1.bold = True
    r_h1.font.size = Pt(13)
    r_h1.font.color.rgb = COLOR_NAVY

    # Paragraph 1: The Context, Network Logic, and Dataset Scope
    p1 = doc.add_paragraph()
    format_spacing(p1, before_pt=0, after_pt=10, line_mult=1.2)
    p1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p1.add_run(
        "Traditional tax audit systems typically evaluate businesses by examining each taxpayer's financial returns in isolation. "
        "However, complex tax avoidance and invoice misuse schemes, such as circular trading rings and fictitious supplier networks, "
        "operate across interconnected webs of buyers and sellers. This study explores how network-based artificial intelligence, "
        "specifically Graph Neural Networks (GNNs), can automatically discover these coordinated patterns directly from trade transaction data "
        "without requiring past audit labels. Utilizing five years of administrative records from Uganda's Electronic Fiscal Receipting and "
        "Invoicing System (EFRIS) between 2018 and 2022 (covering 17,511,132 sales invoice records worth 216.5 trillion UGX and 5,049,092 purchase "
        "invoice records worth 97.8 trillion UGX), we reconstructed an extensive commercial trade network spanning 165,094 registered businesses and "
        "471,167 commercial trade links centered on 614 large corporate taxpayers. By mapping the full commercial connections between buyers "
        "and suppliers, the system evaluates whole supply chains to identify high-risk transactional patterns that standard spreadsheet-style checks routinely miss."
    )

    # Paragraph 2: Findings, High-Exposure Cases, and Practical URA Recommendations
    p2 = doc.add_paragraph()
    format_spacing(p2, before_pt=0, after_pt=14, line_mult=1.2)
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p2.add_run(
        "The multi-model system identified 35 high-priority consensus anomalies that were independently flagged across multiple detection methods, "
        "with 6 entities flagged simultaneously by all models. Within this group, taxpayers exhibiting severe imbalances between claimed purchase tax "
        "credits and reported sales accounted for 3,019.8 billion UGX in invoiced input VAT, offering high-yield candidate leads for targeted tax audits. "
        "In controlled validation tests, combining standard financial indicators with network AI successfully detected 98% of circular trading loops and "
        "97% of shell company pairs, compared to only 20% and 5% using traditional standalone financial filters. For practical tax administration, the study "
        "recommends that the Uganda Revenue Authority adopt a two-tier screening process: an automated baseline filter to instantly catch zero-sales businesses "
        "claiming input tax credits, paired with network AI to uncover hidden supply-chain webs, alongside systematic field audit recording to build Uganda's "
        "first verified training database for future automated compliance systems."
    )

    # -------------------------------------------------------------
    # 4. REPOSITORY & RESEARCH PAPER LINK
    # -------------------------------------------------------------
    p_h2 = doc.add_paragraph()
    format_spacing(p_h2, before_pt=12, after_pt=4, line_mult=1.15)
    r_h2 = p_h2.add_run("Access Full Research Paper & Code Repository")
    r_h2.bold = True
    r_h2.font.size = Pt(12)
    r_h2.font.color.rgb = COLOR_NAVY

    p_link = doc.add_paragraph()
    format_spacing(p_link, before_pt=0, after_pt=14, line_mult=1.15)
    p_link.add_run("• Full Working Paper (PDF): ")
    r_url1 = p_link.add_run("https://github.com/holyasea/GNN-Tax-Anomaly-Uganda/blob/main/paper/main.pdf\n")
    r_url1.bold = True
    r_url1.font.color.rgb = COLOR_LINK
    r_url1.underline = True

    p_link.add_run("• Source Code & Documentation: ")
    r_url2 = p_link.add_run("https://github.com/holyasea/GNN-Tax-Anomaly-Uganda")
    r_url2.bold = True
    r_url2.font.color.rgb = COLOR_LINK
    r_url2.underline = True

    # -------------------------------------------------------------
    # 5. KEY OPERATIONAL TAKEAWAYS (FOR AUDITORS & RISK MANAGERS)
    # -------------------------------------------------------------
    p_h3 = doc.add_paragraph()
    format_spacing(p_h3, before_pt=10, after_pt=6, line_mult=1.15)
    r_h3 = p_h3.add_run("Key Operational Takeaways for URA Teams")
    r_h3.bold = True
    r_h3.font.size = Pt(12)
    r_h3.font.color.rgb = COLOR_NAVY

    takeaways = [
        (
            "1. Move from Isolated Audits to Network Visibility: ",
            "Evaluating taxpayers as connected trade networks rather than standalone files allows URA to spot carousel trading and paper companies that look normal on paper but exist solely to pass fictitious tax credits."
        ),
        (
            "2. Deploy an Immediate Zero-Sales Automated Filter: ",
            "A simple automated rule (Sales = 0 and Purchases > threshold) immediately isolates high-exposure purchase-only filers at zero computational cost, protecting revenue before complex modeling is even run."
        ),
        (
            "3. The 45-Case Blinded Auditor Review Set: ",
            "The multi-model framework identified 35 high-priority consensus cases. To test discrimination and eliminate confirmation bias during URA review, these 35 flagged cases were combined with 10 size-matched non-flagged control cases into a 45-case randomized, blinded review set (auditor_review_blinded.csv) for field audit scoring."
        ),
        (
            "4. Build Uganda's First Ground-Truth Audit Database: ",
            "Systematically logging real-world audit findings (confirmed discrepancies versus normal commercial explanations) will allow URA to transition from unsupervised anomaly detection to highly accurate supervised predictive models."
        )
    ]

    for title, desc in takeaways:
        p_item = doc.add_paragraph()
        format_spacing(p_item, before_pt=2, after_pt=5, line_mult=1.15)
        r_bullet = p_item.add_run("•  ")
        r_bullet.bold = True
        r_bullet.font.color.rgb = COLOR_NAVY
        
        r_t = p_item.add_run(title)
        r_t.bold = True
        r_t.font.color.rgb = COLOR_NAVY
        
        p_item.add_run(desc)

    try:
        doc.save(output_path)
        print(f"Document created successfully at: {output_path}")
    except PermissionError:
        alt_path = output_path.replace(".docx", "_Updated.docx")
        doc.save(alt_path)
        print(f"Primary file was open in Word. Saved updated document to: {alt_path}")

if __name__ == "__main__":
    out = r"D:\Research Project\Project\Project1\paper\URA_Research_Summary_EFRIS_GNN.docx"
    create_ura_summary_document(out)
