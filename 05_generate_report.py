"""
05_generate_report.py
Generiert PDF-Report aus report.md mit eingebetteten Grafiken.
"""

import logging
import markdown
import base64
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "05_generate_report.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def embed_images_in_html(html_content, base_dir):
    """Ersetzt Bild-Pfade durch eingebettete Base64-Daten."""
    def replace_img(match):
        src = match.group(1)
        img_path = base_dir / src
        if img_path.exists():
            with open(img_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return f'src="data:image/png;base64,{data}"'
        logger.warning(f"Bild nicht gefunden: {img_path}")
        return match.group(0)

    return re.sub(r'src="([^"]+)"', replace_img, html_content)


def main():
    md_path = BASE_DIR / "report.md"
    pdf_path = BASE_DIR / "report.pdf"

    logger.info("Lese report.md...")
    md_content = md_path.read_text(encoding="utf-8")

    # Markdown -> HTML
    html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

    # Bilder einbetten
    html_body = embed_images_in_html(html_body, BASE_DIR)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.5;
        max-width: 210mm;
        margin: 15mm auto;
        color: #333;
    }}
    h1 {{ font-size: 18pt; color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 5mm; }}
    h2 {{ font-size: 14pt; color: #34495e; margin-top: 10mm; }}
    h3 {{ font-size: 12pt; color: #555; }}
    h4 {{ font-size: 11pt; color: #666; }}
    table {{ border-collapse: collapse; width: 100%; margin: 5mm 0; font-size: 10pt; }}
    th, td {{ border: 1px solid #ddd; padding: 2mm 3mm; text-align: left; }}
    th {{ background-color: #f5f5f5; font-weight: bold; }}
    img {{ max-width: 100%; height: auto; margin: 3mm 0; }}
    code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 10pt; }}
    @page {{ size: A4; margin: 15mm; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # HTML -> PDF via weasyprint
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(pdf_path))
        logger.info(f"PDF erstellt: {pdf_path}")
    except ImportError:
        # Fallback: HTML speichern
        html_path = BASE_DIR / "report.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info(f"weasyprint nicht verfügbar. HTML gespeichert: {html_path}")
        logger.info("Installiere weasyprint: pip install weasyprint")
    except Exception as e:
        html_path = BASE_DIR / "report.html"
        html_path.write_text(html, encoding="utf-8")
        logger.warning(f"PDF-Erstellung fehlgeschlagen: {e}")
        logger.info(f"HTML gespeichert als Fallback: {html_path}")


if __name__ == "__main__":
    main()
