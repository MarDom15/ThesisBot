from fpdf import FPDF

def export_pdf(text, filename="output.pdf", title="Synthèse de Thèse", author="Thesis Assistant AI"):
    """
    Exporte un texte en PDF avec titre et auteur.

    Paramètres :
    - text : le contenu texte à exporter
    - filename : nom du fichier PDF généré
    - title : titre affiché en haut du PDF
    - author : auteur affiché en haut du PDF
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Titre
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(5)

    # Auteur
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, f"Author: {author}", ln=True, align='C')
    pdf.ln(10)

    # Contenu texte
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf.output(filename)

