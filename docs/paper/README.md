# Building and extending the article

The article is written in English, consistently with the repository. Its source is [index.rst](index.rst), which includes the files in `sections/`. Mathematical expressions use LaTeX syntax inside Sphinx math directives. **Edit these sources, not the generated `.tex` file.**

Sphinx's [LaTeX builder](https://www.sphinx-doc.org/en/master/usage/builders/index.html#sphinx.builders.latex.LaTeXBuilder) produces an article document and its supporting files. Tectonic then compiles that LaTeX to PDF. The same source also builds as HTML.

## Build

From the repository root:

```bash
.venv/bin/python -m pip install -e '.[docs]'
make article-html
make article-latex
make article
```

| Command | Output | Requirements |
| --- | --- | --- |
| `make article-html` | `docs/_build/html/index.html` | Sphinx and Matplotlib |
| `make article-latex` | `docs/_build/latex/vela-belief.tex` and supporting files | Sphinx and Matplotlib |
| `make article` | `docs/_build/latex/vela-belief.pdf` | The above, plus Tectonic |

The Makefile uses `.venv/bin/python`. Override it with `PYTHON=python` if using another environment. Sphinx warnings fail the build. CI checks HTML and LaTeX generation; the PDF must also be compiled when changing article layout or LaTeX configuration.

Install [Tectonic](https://tectonic-typesetting.github.io/en-US/install.html) on your `PATH`, or place its binary at `.tools/tectonic`. An alternative location can be passed as `make article TECTONIC=/absolute/path/to/tectonic`. The local setup was verified with Sphinx 8.2.3 and Tectonic 0.17.0, using the official Linux x86-64 musl release. Its archive SHA-256 is:

```text
8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7
```

The first PDF build downloads required TeX resources into `.cache/tectonic`; later builds reuse them. Compiler binaries, caches, generated figures, and build outputs are excluded from Git. HTML/LaTeX generation itself requires no network or training run. HTML math uses Sphinx's default MathJax renderer.

If a full TeX Live installation with XeLaTeX and latexmk is already available, the standard Sphinx PDF command can be used instead:

```bash
.venv/bin/python -m sphinx -M latexpdf docs/paper docs/_build -W --keep-going -n
```

## Source structure

| File | Purpose |
| --- | --- |
| `sections/abstract.rst` | Current contribution and manuscript status |
| `sections/introduction.rst` | Motivation, scope, and selected related work |
| `sections/method.rst` | Belief definition, model, and training objective |
| `sections/stage1.rst` | HMM protocol, results, tests, and reproducibility |
| `sections/roadmap.rst` | Prospective methods and evaluation for stages 2–6 |
| `sections/discussion.rst` | Limitations and current conclusion |
| `sections/references.rst` | Cited sources |
| `results.py` | Generate tables and a vector figure from `../stage1-results.json` |
| `conf.py` | Sphinx settings and LaTeX article layout |

The figure and tables are rebuilt automatically in `_generated/`. Their source snapshot hash is included in the article. The numerical snapshot remains versioned and is not silently replaced with whatever local run happens to exist.

## Extend after each experimental stage

1. Complete the experiment and its checks; archive its resolved configurations, metrics, seeds, and software versions.
2. Add a dedicated `sections/stageN.rst` describing the question, method, comparison budget, results, and limitations. Include it from `index.rst` after the preceding stage.
3. Generate its tables and figures from its archived numerical data. Add references for newly discussed methods.
4. Remove the completed stage's prospective description from `roadmap.rst`. Update the abstract and discussion to reflect the evidence, including null results.
5. Run `make check`, `make article-html`, and `make article`; inspect the PDF's equations, tables, figure, and references.

After stage 6, replace the remaining roadmap with a synthesis across stages and finalize the author list, venue format, and reproducibility materials. Preserve the distinction between completed experiments and any future LLM or world-model application.
