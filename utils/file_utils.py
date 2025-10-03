from fpdf import FPDF

def export_pdf(text, filename="output.pdf", title="Thesis Summary", author="Thesis Assistant AI"):
    """
    Export text to a PDF with title and author.

    Parameters:
    - text : the text content to export
    - filename : the name of the generated PDF file
    - title : title displayed at the top of the PDF
    - author : author displayed at the top of the PDF
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(5)

    # Author
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, f"Author: {author}", ln=True, align='C')
    pdf.ln(10)

    # Text content
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf.output(filename)
