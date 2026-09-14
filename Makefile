PYTHON ?= .venv/bin/python
SPHINX = $(PYTHON) -m sphinx
SPHINXOPTS = -W --keep-going -n
TECTONIC ?= $(if $(wildcard .tools/tectonic),$(abspath .tools/tectonic),tectonic)

.PHONY: check article article-html article-latex

check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m pytest -q

article-html:
	$(SPHINX) -M html docs/paper docs/_build $(SPHINXOPTS)

article-latex:
	$(SPHINX) -M latex docs/paper docs/_build $(SPHINXOPTS)

article: article-latex
	cd docs/_build/latex && TECTONIC_CACHE_DIR="$(CURDIR)/.cache/tectonic" "$(TECTONIC)" --keep-logs vela-belief.tex
