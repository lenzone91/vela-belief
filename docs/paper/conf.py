"""One article source, rendered as HTML or a Sphinx LaTeX article."""

import os
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SOURCE.parents[1] / ".cache" / "matplotlib"))
sys.path.insert(0, str(SOURCE))

project = "VELA-Belief: Toward Bounded, Structured and Evolving Latent Memory"
author = "VELA project"
copyright = "2026, VELA project"
release = "Research design and stage 1 results"
today = "September 2026 — research design and stage 1 results"
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
# Sphinx 8's colorrows hooks recurse with TeX Live 2026's colortbl/array.
# https://github.com/sphinx-doc/sphinx/issues/14465
latex_table_style = ["booktabs"]
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
    "tableofcontents": r"\setcounter{tocdepth}{2}\tableofcontents\clearpage",
}


def generate_results(app):
    from diagrams import generate as generate_diagrams
    from results import generate

    generate(Path(app.srcdir))
    generate_diagrams(Path(app.srcdir))


def setup(app):
    from overleaf import package_overleaf

    app.connect("builder-inited", generate_results)
    app.connect("build-finished", package_overleaf)
