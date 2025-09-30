from fpdf import FPDF

def export_pdf(text, filename="output.pdf"):
    """
    Exporte un texte en PDF.

    Paramètres :
    - text : le contenu texte à exporter
    - filename : nom du fichier PDF généré
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Ajout du texte avec multi_cell pour le retour à la ligne
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf.output(filename)

