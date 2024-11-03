# utils/export.py
from fpdf import FPDF
from docx import Document
import tempfile

class DocumentExporter:
    @staticmethod
    def export_to_pdf(text: str, output_path: str):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Split text into lines that fit within page width
        lines = text.split('\n')
        for line in lines:
            pdf.multi_cell(0, 10, line)
            
        pdf.output(output_path)

    @staticmethod
    def export_to_docx(text: str, output_path: str):
        doc = Document()
        doc.add_paragraph(text)
        doc.save(output_path)

