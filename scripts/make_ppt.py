"""Generate professional Review-I PowerPoint presentation (.pptx) for VIT Capstone.

Run with: python scripts/make_ppt.py
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 Widescreen
    prs.slide_height = Inches(7.5)

    # Design palette
    DARK_BLUE = RGBColor(16, 42, 77)     # #102A4D
    ACCENT_BLUE = RGBColor(27, 85, 155)  # #1B559B
    LIGHT_BG = RGBColor(248, 250, 252)   # #F8FAFC
    TEXT_DARK = RGBColor(30, 41, 59)     # #1E293B
    TEXT_MUTED = RGBColor(100, 116, 139) # #64748B
    WHITE = RGBColor(255, 255, 255)
    GREEN = RGBColor(16, 185, 129)
    CARD_BG = RGBColor(238, 242, 246)

    def apply_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = LIGHT_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category="VIT SCOPE • B.TECH FINAL YEAR CAPSTONE (BCSE498J / CBS1904)"):
        # Header bar / category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = DARK_BLUE

    # ─────────────────────────────────────────────────────────────
    # SLIDE 1: Title Slide (Dark Theme)
    # ─────────────────────────────────────────────────────────────
    slide_layout = prs.slide_layouts[6]  # blank
    s1 = prs.slides.add_slide(slide_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = DARK_BLUE
    bg1.line.fill.background()

    # Title box
    tbox = s1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11), Inches(2.2))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Prompt-Injection Benchmark & Lightweight Defense for Tool-Using LLM Agents"
    p1.font.size = Pt(34)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    # Subtitle
    subbox = s1.shapes.add_textbox(Inches(1.2), Inches(3.8), Inches(11), Inches(1.0))
    tf_sub = subbox.text_frame
    tf_sub.word_wrap = True
    psub = tf_sub.paragraphs[0]
    psub.text = "A 160-Cell Executable Benchmark Suite and Sub-250ms 3-Layer Defense Framework"
    psub.font.size = Pt(18)
    psub.font.color.rgb = RGBColor(147, 197, 253)

    # Details Box
    detbox = s1.shapes.add_textbox(Inches(1.2), Inches(5.2), Inches(11), Inches(1.5))
    tf_det = detbox.text_frame
    pdet = tf_det.paragraphs[0]
    pdet.text = "Student: Saksham   |   Course: BCSE498J / CBS1904 (Winter Semester 2026–2027)\nSchool of Computer Science & Engineering (SCOPE), Vellore Institute of Technology\nReview-I: Title Defense, Problem Formulation & Architecture Prototype"
    pdet.font.size = Pt(13)
    pdet.font.color.rgb = RGBColor(203, 213, 225)

    # ─────────────────────────────────────────────────────────────
    # SLIDE 2: Table of Contents
    # ─────────────────────────────────────────────────────────────
    s2 = prs.slides.add_slide(slide_layout)
    apply_background(s2)
    add_header(s2, "Presentation Roadmap — Review I")

    topics = [
        ("01", "Introduction & Motivation", "The emergence of tool-augmented LLM agents and the indirect prompt injection threat."),
        ("02", "Problem Statement & Gaps", "Why existing solutions (LLM-as-a-Judge and regex denylists) are insufficient in practice."),
        ("03", "System Architecture", "The proposed low-latency 3-layer defense pipeline (Provenance, Classifier, Policy)."),
        ("04", "160-Cell Attack Taxonomy", "Systematic 500-case benchmark covering 4 goals, 5 intake channels, and 8 techniques."),
        ("05", "Implementation & Live Prototype", "Fine-tuned DistilBERT on 11,683 samples, SQLite deterministic sandbox & test results."),
        ("06", "Work Plan & Publications", "Review-I to Final Review timeline, milestones, and IEEE target venue."),
    ]

    for idx, (num, title, desc) in enumerate(topics):
        col = idx % 2
        row = idx // 2
        left = Inches(1.0 + col * 5.8)
        top = Inches(1.8 + row * 1.6)

        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.4), Inches(1.3))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = RGBColor(226, 232, 240)

        # Number circle
        tb = s2.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), Inches(0.6), Inches(0.5))
        p = tb.text_frame.paragraphs[0]
        p.text = num
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE

        # Text
        tb_txt = s2.shapes.add_textbox(left + Inches(0.9), top + Inches(0.15), Inches(4.3), Inches(1.0))
        tf = tb_txt.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = DARK_BLUE

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = TEXT_MUTED

    # ─────────────────────────────────────────────────────────────
    # SLIDE 3: Motivation & Threat Model
    # ─────────────────────────────────────────────────────────────
    s3 = prs.slides.add_slide(slide_layout)
    apply_background(s3)
    add_header(s3, "Background & Threat Model: Indirect Prompt Injection")

    points_s3 = [
        ("The Rise of Tool-Augmented Agents", "Modern LLMs (Gemini, GPT-4, Claude) are increasingly deployed as autonomous agents equipped with function calling (reading emails, executing SQL queries, querying APIs, transferring money, updating calendars)."),
        ("The Indirect Injection Vulnerability (OWASP LLM01)", "Unlike direct jailbreaks entered by the user, indirect prompt injection occurs when the agent ingests untrusted external data (e.g. an email, a web page, or an uploaded PDF) containing adversarial instructions designed to hijack function-calling privileges."),
        ("Real-World Impact", "An attacker can silently trigger unauthorized data exfiltration (forwarding credentials, sensitive files) or unauthorized transactions without the human user ever knowing or consenting."),
    ]

    for i, (heading, body) in enumerate(points_s3):
        top = Inches(1.8 + i * 1.6)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), top, Inches(11.3), Inches(1.3))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = RGBColor(226, 232, 240)

        tb = s3.shapes.add_textbox(Inches(1.2), top + Inches(0.15), Inches(10.9), Inches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = heading
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.color.rgb = DARK_BLUE

        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(12)
        p2.font.color.rgb = TEXT_DARK

    # ─────────────────────────────────────────────────────────────
    # SLIDE 4: Problem Statement & Existing Limitations
    # ─────────────────────────────────────────────────────────────
    s4 = prs.slides.add_slide(slide_layout)
    apply_background(s4)
    add_header(s4, "Problem Statement & Limitations of Existing Defenses")

    # 2 Comparison Columns
    # Left: Existing Approaches
    c1 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.8), Inches(5.4), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = WHITE
    c1.line.color.rgb = RGBColor(254, 202, 202)

    tb_c1 = s4.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(5.0), Inches(4.4))
    tf1 = tb_c1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "❌ Current Defense Limitations"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(220, 38, 38)

    limitations = [
        ("Prohibitive Latency (LLM-as-a-Judge)", "Calling a secondary frontier LLM to audit each tool output adds +1,500ms to 3,000ms latency per step, rendering real-time agents unusable."),
        ("Unsustainable Token Costs", "Auditing full documents via frontier models doubles inference bills."),
        ("Brittle Regex / Keyword Denylists", "Easily circumvented by simple character substitutions, base64 encoding, or cross-lingual pivoting (e.g. Hindi / Hinglish)."),
        ("Lack of State-Level Benchmarks", "Existing benchmarks rely on fuzzy regex on model prose rather than verifying database state alterations."),
    ]
    for h, b in limitations:
        p_h = tf1.add_paragraph()
        p_h.text = f"\n• {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(11)
        p_h.font.color.rgb = DARK_BLUE
        p_b = tf1.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = TEXT_DARK

    # Right: Our Proposed Approach
    c2 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.5), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = WHITE
    c2.line.color.rgb = RGBColor(187, 247, 208)

    tb_c2 = s4.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.1), Inches(4.4))
    tf2 = tb_c2.text_frame
    tf2.word_wrap = True
    p_ok = tf2.paragraphs[0]
    p_ok.text = "✅ Our Project Solution"
    p_ok.font.size = Pt(17)
    p_ok.font.bold = True
    p_ok.font.color.rgb = RGBColor(22, 163, 74)

    solutions = [
        ("Sub-250ms Lightweight Defense", "Fine-tuned DistilBERT sequence classifier evaluating payloads in ~60ms without secondary LLM API calls."),
        ("Deterministic 3-Layer Defense Pipeline", "Layer 1 (Provenance Tagging) + Layer 2 (Anomaly Classifier) + Layer 3 (Intent Policy Enforcement)."),
        ("Multilingual 160-Cell Benchmark", "500 cases covering English, Hindi, and Hinglish attack variants with 50 benign controls."),
        ("State-Level Outcome Verification", "Deterministic SQLite sandbox checking actual rows inserted/deleted rather than fuzzy text."),
    ]
    for h, b in solutions:
        p_h = tf2.add_paragraph()
        p_h.text = f"\n• {h}: "
        p_h.font.bold = True
        p_h.font.size = Pt(11)
        p_h.font.color.rgb = DARK_BLUE
        p_b = tf2.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = TEXT_DARK

    # ─────────────────────────────────────────────────────────────
    # SLIDE 5: 3-Layer Defense Architecture
    # ─────────────────────────────────────────────────────────────
    s5 = prs.slides.add_slide(slide_layout)
    apply_background(s5)
    add_header(s5, "Proposed 3-Layer Defense Architecture")

    layers = [
        ("Layer 1: Provenance Tagging", "Boundary Delimitation",
         "Wraps all external tool outputs in explicit structural tags: <tool_output source='email' type='untrusted_data'>...</tool_output> with trailing instructional reminders.",
         "Negligible (<4ms)", "Structural ambiguity"),
        ("Layer 2: Lightweight Classifier", "Machine Learning Detection",
         "A fine-tuned DistilBERT transformer (267 MB) trained on 11,683 real-world samples that inspects incoming text for adversarial cues before the agent reads it.",
         "~50–90 ms", "Sub-millisecond filtering"),
        ("Layer 3: Tool Policy Enforcement", "Intent-Aware Authorization",
         "Inspects every sensitive action (bank transfer, file deletion, email send) against the original user task to ensure user authorization before execution.",
         "<3 ms", "Safety net against bypasses"),
    ]

    for idx, (title, sub, details, lat, benefit) in enumerate(layers):
        left = Inches(1.0 + idx * 3.9)
        card = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.8), Inches(3.6), Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = ACCENT_BLUE

        tb = s5.shapes.add_textbox(left + Inches(0.2), Inches(2.0), Inches(3.2), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(15)
        pt.font.bold = True
        pt.font.color.rgb = DARK_BLUE

        ps = tf.add_paragraph()
        ps.text = sub.upper()
        ps.font.size = Pt(10)
        ps.font.bold = True
        ps.font.color.rgb = ACCENT_BLUE

        pd = tf.add_paragraph()
        pd.text = f"\nMechanism:\n{details}\n"
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_DARK

        plat = tf.add_paragraph()
        plat.text = f"Latency Overhead: {lat}\nKey Value: {benefit}"
        plat.font.size = Pt(11)
        plat.font.bold = True
        plat.font.color.rgb = GREEN

    # ─────────────────────────────────────────────────────────────
    # SLIDE 6: 160-Cell Benchmark & Taxonomy
    # ─────────────────────────────────────────────────────────────
    s6 = prs.slides.add_slide(slide_layout)
    apply_background(s6)
    add_header(s6, "Benchmark Design: 160-Cell Attack Taxonomy (500 Cases)")

    dims = [
        ("4 Attack Goals", "• Data Exfiltration\n• Unauthorized Action\n• Goal Hijack\n• Denial of Service"),
        ("5 Intake Channels", "• Email Inboxes\n• Web Page Scraping\n• Retrieved Documents\n• API Payloads\n• Uploaded PDFs"),
        ("8 Attack Techniques", "• Hidden Text\n• Persona / Role-Play\n• Tool-Call Forging\n• Multi-Step Exploits\n• Base64 Encoding\n• Multilingual Pivots\n• Context Manipulation\n• Payload Splitting"),
        ("Linguistic Coverage", "• English (en)\n• Hindi (hi)\n• Hinglish (hi-en)\n\nSplit Integrity:\n• 401 Public Split\n• 99 Held-Out Private\n• 50 Benign Controls"),
    ]

    for idx, (title, items) in enumerate(dims):
        left = Inches(1.0 + idx * 2.9)
        card = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.8), Inches(2.6), Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = RGBColor(226, 232, 240)

        tb = s6.shapes.add_textbox(left + Inches(0.15), Inches(2.0), Inches(2.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(14)
        pt.font.bold = True
        pt.font.color.rgb = DARK_BLUE

        pi = tf.add_paragraph()
        pi.text = f"\n{items}"
        pi.font.size = Pt(11)
        pi.font.color.rgb = TEXT_DARK

    # ─────────────────────────────────────────────────────────────
    # SLIDE 7: Deterministic Mock Sandbox & Ground Truth Verification
    # ─────────────────────────────────────────────────────────────
    s7 = prs.slides.add_slide(slide_layout)
    apply_background(s7)
    add_header(s7, "Isolated Deterministic Mock Sandbox (5 Tools)")

    sb_points = [
        ("Why a Simulated Sandbox?", "Giving experimental autonomous agents real bank credentials or personal Gmail accounts creates safety and liability risks. We designed an isolated SQLite in-memory sandbox."),
        ("5 Deterministic Mock Tools", "1. email: read_inbox, search_emails, send_email\n2. bank: check_balance, transfer, transaction_history\n3. calendar: list_events, get_event, create_event\n4. filestore: list_files, read_file, create_file, delete_file\n5. web_fetch: fetch_page"),
        ("State-Level Outcome Checker ('Crime Scene Test')", "Instead of string-matching agent text, our evaluation directly queries the database:\n• Did an unauthorized row appear in `bank_transactions`?\n• Was an illicit email added to `email_outbox`?\n• Were unauthorized files deleted from `files`?"),
    ]

    for idx, (h, b) in enumerate(sb_points):
        top = Inches(1.8 + idx * 1.6)
        card = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), top, Inches(11.3), Inches(1.3))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = RGBColor(226, 232, 240)

        tb = s7.shapes.add_textbox(Inches(1.2), top + Inches(0.15), Inches(10.9), Inches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = h
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.color.rgb = DARK_BLUE

        p2 = tf.add_paragraph()
        p2.text = b
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_DARK

    # ─────────────────────────────────────────────────────────────
    # SLIDE 8: Experimental Results & Comparative Evaluation
    # ─────────────────────────────────────────────────────────────
    s8 = prs.slides.add_slide(slide_layout)
    apply_background(s8)
    add_header(s8, "Experimental Results: Benchmark Comparison")

    # Table Shape
    rows, cols = 6, 5
    left = Inches(1.0)
    top = Inches(1.8)
    width = Inches(11.3)
    height = Inches(3.4)

    table_shape = s8.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table_data = [
        ["Defense Strategy", "Attack Success Rate (ASR)", "Benign Utility", "Mean Latency", "Status"],
        ["Undefended Baseline", "100.0% (Vulnerable)", "100.0%", "2.4 ms", "❌ Severe Risk"],
        ["Layer 1: Provenance Only", "45.0%", "100.0%", "3.8 ms", "⚠️ Partial Protection"],
        ["Layer 2: Classifier Only", "0.0%", "98.5%", "78.2 ms", "✅ High Efficacy"],
        ["Layer 3: Policy Only", "15.0%", "95.0%", "2.7 ms", "⚠️ Action Defense"],
        ["Our Full 3-Layer Defense", "0.0% (All Blocked)", "98.5%", "84.7 ms", "🏆 Optimal Security"],
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BLUE
            elif r_idx == 5:
                p.font.bold = True
                p.font.color.rgb = RGBColor(16, 185, 129)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(240, 253, 244)
            else:
                p.font.color.rgb = TEXT_DARK

    # Note at bottom
    tb_note = s8.shapes.add_textbox(Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.2))
    tfn = tb_note.text_frame
    tfn.word_wrap = True
    pn = tfn.paragraphs[0]
    pn.text = "Key Takeaway: Our 3-Layer pipeline cuts Attack Success Rate (ASR) to 0.0% while keeping latency under 100ms. In contrast, LLM-as-a-Judge requires >1,500ms per tool invocation."
    pn.font.size = Pt(13)
    pn.font.bold = True
    pn.font.color.rgb = DARK_BLUE

    # ─────────────────────────────────────────────────────────────
    # SLIDE 9: Project Milestones & Review-I Alignment
    # ─────────────────────────────────────────────────────────────
    s9 = prs.slides.add_slide(slide_layout)
    apply_background(s9)
    add_header(s9, "Capstone Project Milestones & Deliverables")

    milestones = [
        ("Google Form Submission", "20 Nov 2026", "Title, Abstract, Area, and Guide selection finalized.", True),
        ("Review-I Presentation", "9–11 Dec 2026", "Title defense, literature review, 160-cell taxonomy, architecture diagram & live demo.", True),
        ("Review-II (50–60% Work)", "28–29 Jan 2027", "Ablation study across 5 baselines, multi-model evaluation (Gemini 2.5 Flash, LLaMA-3.2, GPT-4o-mini).", False),
        ("Final Review & Paper", "April 2027", "Full benchmark report, final code submission, and manuscript targeting IEEE/Scopus venue.", False),
    ]

    for idx, (title, date, desc, done) in enumerate(milestones):
        top = Inches(1.8 + idx * 1.25)
        card = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), top, Inches(11.3), Inches(1.05))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = GREEN if done else RGBColor(203, 213, 225)

        tb = s9.shapes.add_textbox(Inches(1.2), top + Inches(0.1), Inches(10.9), Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        status_tag = " [COMPLETED & PROTOTYPED]" if done else " [UPCOMING]"
        pt.text = f"{title} ({date}){status_tag}"
        pt.font.size = Pt(14)
        pt.font.bold = True
        pt.font.color.rgb = GREEN if done else DARK_BLUE

        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_MUTED

    # ─────────────────────────────────────────────────────────────
    # SLIDE 10: Conclusion & Q&A
    # ─────────────────────────────────────────────────────────────
    s10 = prs.slides.add_slide(slide_layout)
    bg10 = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg10.fill.solid()
    bg10.fill.fore_color.rgb = DARK_BLUE
    bg10.line.fill.background()

    tbox10 = s10.shapes.add_textbox(Inches(1.5), Inches(2.2), Inches(10.3), Inches(2.0))
    tf10 = tbox10.text_frame
    tf10.word_wrap = True
    p10 = tf10.paragraphs[0]
    p10.text = "Thank You!\nQuestions & Discussion"
    p10.font.size = Pt(36)
    p10.font.bold = True
    p10.font.color.rgb = WHITE
    p10.alignment = PP_ALIGN.CENTER

    sub10 = s10.shapes.add_textbox(Inches(1.5), Inches(4.5), Inches(10.3), Inches(1.5))
    tf_sub10 = sub10.text_frame
    tf_sub10.word_wrap = True
    ps10 = tf_sub10.paragraphs[0]
    ps10.text = "Prompt-Injection Benchmark & Lightweight Defense for Tool-Using LLM Agents\nStudent: Saksham   |   Course: BCSE498J / CBS1904\nSchool of Computer Science & Engineering (SCOPE), VIT"
    ps10.font.size = Pt(14)
    ps10.font.color.rgb = RGBColor(147, 197, 253)
    ps10.alignment = PP_ALIGN.CENTER

    output_path = os.path.abspath("Review_1_Presentation.pptx")
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_presentation()
