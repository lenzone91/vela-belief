"""One article source, rendered as HTML or a Sphinx LaTeX article."""

import os
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SOURCE.parents[1] / ".cache" / "matplotlib"))
sys.path.insert(0, str(SOURCE))

project = "VELA-Belief: Learning Persistent Latent Belief States"
author = "VELA project"
copyright = "2026, VELA project"
release = "Stage 1 working draft"
today = "Stage 1 working draft"
language = "en"
root_doc = "index"
exclude_patterns = ["sections/**", "_generated/**"]
extensions = []
numfig = True
html_theme = "alabaster"
html_title = project
html_show_sourcelink = False

latex_engine = "xelatex"
latex_documents = [("index", "vela-belief.tex", project, author, "howto")]
latex_domain_indices = False
latex_show_urls = "footnote"
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "extraclassoptions": "oneside",
    "fontpkg": r"""\setmainfont{lmroman10-regular.otf}[
  BoldFont=lmroman10-bold.otf, ItalicFont=lmroman10-italic.otf,
  BoldItalicFont=lmroman10-bolditalic.otf]
\setsansfont{lmsans10-regular.otf}[
  BoldFont=lmsans10-bold.otf, ItalicFont=lmsans10-oblique.otf,
  BoldItalicFont=lmsans10-boldoblique.otf]
\setmonofont{lmmono10-regular.otf}[ItalicFont=lmmono10-italic.otf]
""",
    "sphinxsetup": "iconpackage=none",
    "preamble": r"\usepackage{microtype}",
    "maketitle": r"\maketitle",
    "tableofcontents": "",
}


def generate_results(app):
    from results import generate

    generate(Path(app.srcdir))


def setup(app):
    app.connect("builder-inited", generate_results)
