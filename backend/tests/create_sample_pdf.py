
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

pdf_path = "sample_documents/company_policy.pdf"

doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()

content = [
    Paragraph("Company Employee Policy", styles["Title"]),
    Spacer(1, 20),

    Paragraph("1. Leave Policy", styles["Heading2"]),
    Paragraph(
        "Every employee receives 12 casual leaves and "
        "12 sick leaves per calendar year.",
        styles["BodyText"]
    ),
    Spacer(1, 12),

    Paragraph("2. Work Timings", styles["Heading2"]),
    Paragraph(
        "Standard working hours are 9 AM to 6 PM, "
        "Monday to Friday. Employees receive a one-hour lunch break.",
        styles["BodyText"]
    ),
    Spacer(1, 12),

    Paragraph("3. Work From Home", styles["Heading2"]),
    Paragraph(
        "Employees can request up to two work-from-home "
        "days per month with manager approval.",
        styles["BodyText"]
    ),
    Spacer(1, 12),

    Paragraph("4. Salary Policy", styles["Heading2"]),
    Paragraph(
        "Salaries are credited on the last working day "
        "of every month.",
        styles["BodyText"]
    ),
]

doc.build(content)

print("Sample PDF created successfully!")
