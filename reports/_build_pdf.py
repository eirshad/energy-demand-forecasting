"""Render reports/REPORT.md to a styled reports/REPORT.pdf (via HTML + headless Chrome).

    uv run --with markdown python reports/_build_pdf.py
"""

import pathlib
import subprocess

import markdown

HERE = pathlib.Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.5; color: #1d2433; max-width: 100%; }
h1 { font-size: 22pt; color: #0f2b4c; margin: 0 0 4pt; line-height: 1.2; }
h1 + p em { color: #5b6577; }
h2 { font-size: 15pt; color: #0f2b4c; border-bottom: 2px solid #d5deea; padding-bottom: 3pt;
     margin-top: 22pt; break-after: avoid; }
h3 { font-size: 12pt; color: #1f4e79; margin-top: 16pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
strong { color: #0f2b4c; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.8pt; background: #f1f4f8;
       padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fb; border: 1px solid #e1e7ef; border-radius: 6px; padding: 8pt 10pt;
      font-size: 8.5pt; line-height: 1.4; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9.2pt;
        break-inside: avoid; }
th { background: #1f4e79; color: #fff; text-align: left; padding: 5pt 7pt; font-weight: 600; }
td { padding: 4pt 7pt; border-bottom: 1px solid #e3e8ef; vertical-align: top; }
tr:nth-child(even) td { background: #f6f8fb; }
img { display: block; max-width: 100%; margin: 10pt auto 2pt; break-inside: avoid; }
p:has(> img) { break-inside: avoid; break-after: avoid; margin-bottom: 0; }
p > em:only-child { display: block; text-align: center; font-size: 8.8pt; color: #5b6577;
                    margin-bottom: 10pt; }
hr { border: none; border-top: 1px solid #d5deea; margin: 16pt 0; }
ul, ol { padding-left: 18pt; }
li { margin: 2pt 0; }
"""


def main() -> None:
    md_text = (HERE / "REPORT.md").read_text()
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Day-ahead electricity demand forecasting for Germany</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    html_path = HERE / "_report_print.html"
    html_path.write_text(html)
    pdf_path = HERE / "REPORT.pdf"
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--allow-file-access-from-files",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    html_path.unlink()
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
