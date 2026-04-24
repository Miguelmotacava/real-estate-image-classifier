"""Convierte reports/informe/informe.md a informe.pdf usando markdown-pdf.

Uso:
    python reports/informe/build_pdf.py
"""
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

HERE = Path(__file__).resolve().parent
SRC = HERE / "informe.md"
OUT = HERE / "informe.pdf"

CSS = """
body { font-family: 'Segoe UI', 'Helvetica', sans-serif; font-size: 10.5pt; line-height: 1.4; color: #222; }
h1 { font-size: 18pt; margin-top: 0.2em; border-bottom: 2px solid #444; padding-bottom: 4px; }
h2 { font-size: 14pt; margin-top: 1.1em; color: #1a1a1a; }
h3 { font-size: 12pt; margin-top: 0.9em; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 0.7em 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: left; }
th { background: #eee; }
code { font-family: 'Consolas', 'Courier New', monospace; font-size: 9.5pt; background: #f3f3f3; padding: 1px 3px; }
pre { background: #f3f3f3; padding: 8px; border-radius: 4px; font-size: 9pt; line-height: 1.25; overflow-x: auto; }
a { color: #0b5ea8; text-decoration: none; }
strong { color: #000; }
"""


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    pdf = MarkdownPdf(toc_level=0, optimize=True)
    pdf.add_section(Section(md_text, paper_size="A4"), user_css=CSS)
    pdf.meta["title"] = "Clasificador de imágenes inmobiliarias"
    pdf.meta["author"] = "Pedro Calderón, Juan Miguel Correa, Miguel Mota Cava"
    pdf.meta["subject"] = "Machine Learning II — Máster en Big Data"
    pdf.save(str(OUT))
    print(f"OK: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
