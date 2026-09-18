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
| `make article-latex` | `docs/_build/latex/vela-belief.tex`, supporting files, and `docs/_build/vela-belief-overleaf.zip` | Sphinx and Matplotlib |
| `make article` | `docs/_build/latex/vela-belief.pdf` | The above, plus Tectonic |

The Makefile uses `.venv/bin/python`. Override it with `PYTHON=python` if using another environment. Sphinx warnings fail the build. CI checks HTML and LaTeX generation; the PDF must also be compiled when changing article layout or LaTeX configuration.

With the project's existing Mamba environment:

```bash
mamba activate vela
make article-html article-latex PYTHON=python
make article PYTHON=python
```

Install [Tectonic](https://tectonic-typesetting.github.io/en-US/install.html) on your `PATH`, or place its binary at `.tools/tectonic`. An alternative location can be passed as `make article TECTONIC=/absolute/path/to/tectonic`. The local setup was verified with Sphinx 8.2.3 and Tectonic 0.17.0, using the official Linux x86-64 musl release. Its archive SHA-256 is:

```text
8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7
```

The first PDF build downloads required TeX resources into `.cache/tectonic`; later builds reuse them. Compiler binaries, caches, generated figures, and build outputs are excluded from Git. HTML/LaTeX generation itself requires no network or training run. HTML math uses Sphinx's default MathJax renderer.

If a full TeX Live installation with XeLaTeX and latexmk is already available, the standard Sphinx PDF command can be used instead:

```bash
.venv/bin/python -m sphinx -M latexpdf docs/paper docs/_build -W --keep-going -n
```

## Import into Overleaf

Run `make article-latex` (or `make article-latex PYTHON=python` in the Mamba
environment). The build also creates `docs/_build/vela-belief-overleaf.zip`.

1. In Overleaf, select **New Project > Upload Project** and upload that ZIP.
2. Set **Main document** to `vela-belief.tex` and **Compiler** to **XeLaTeX**
   in the project settings.
3. Recompile.

The archive places the LaTeX source, Sphinx styles, `latexmkrc`, and PDF figures
at the project root. It excludes local compilation outputs, including the
article PDF and logs. Overleaf needs no Python, Sphinx, or Tectonic setup.
See Overleaf's [project upload instructions](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project).

Tables use `booktabs` without alternating row colours: Sphinx 8's `colorrows`
style is incompatible with the updated `colortbl`/`array` packages in TeX Live
2026 ([Sphinx issue #14465](https://github.com/sphinx-doc/sphinx/issues/14465)).
If an older export fails with `TeX capacity exceeded` at `\sphinxmidrule`,
upload a freshly generated archive and use **Recompile from scratch**.

Keep permanent edits in the repository's RST sources and regenerate the archive;
changes made to the generated LaTeX in Overleaf are not imported back into RST.

## Source structure

| File | Purpose |
| --- | --- |
| `sections/abstract.rst` | Current contribution and manuscript status |
| `sections/introduction.rst` | Problem, running example, design progression, and claim status |
| `sections/related_work.rst` | Original-paper comparisons and intended contribution |
| `sections/foundations.rst` | Notation, belief sufficiency, predictive compression, and capacity |
| `sections/memory.rst` | Routing, update, allocation, consolidation, forgetting, and pseudocode |
| `sections/objectives.rst` | Learning pressures, optimization, and failure modes |
| `sections/llm.rst` | Overflow, VELA-S, frozen-reader distillation, cache consistency, VELA-KV, episodic memory |
| `sections/world_models.rst` | Filtering versus imagination, stochastic state, continuous references, and planning |
| `sections/method.rst` | Implemented GRU baseline and its training objective |
| `sections/stage1.rst` | HMM protocol, results, tests, and reproducibility |
| `sections/evaluation.rst` | Hypotheses, tasks, baselines, resource accounting, and statistical protocol |
| `sections/roadmap.rst` | Nine stages, dependencies, comparisons, deliverables, and decision criteria |
| `sections/discussion.rst` | Limitations and current conclusion |
| `sections/glossary.rst` | Definitions and coverage of the 40 topics in `patch.md` |
| `sections/references.rst` | Cited sources |
| `results.py` | Generate tables and a vector figure from `../stage1-results.json` |
| `diagrams.py` | Generate four original conceptual diagrams as SVG and PDF |
| `conf.py` | Sphinx settings and LaTeX article layout |
| `overleaf.py` | Package LaTeX sources and figures into an Overleaf import ZIP |

The figure and tables are rebuilt automatically in `_generated/`. Their source snapshot hash is included in the article. The numerical snapshot remains versioned and is not silently replaced with whatever local run happens to exist.

The memory lifecycle, consolidation, LLM overflow, and roadmap illustrations are also rebuilt automatically. They are conceptual drawings, not measured results. Edit `diagrams.py` to change them; no diagram service, image-generation dependency, or network request is needed. Sphinx selects SVG for HTML and PDF for LaTeX. Both formats include a reading map/table of contents; equations, figure captions, and internal references share the same source.

## Extend after each experimental stage

1. Complete the experiment and its checks; archive its resolved configurations, metrics, seeds, and software versions.
2. Add a dedicated `sections/stageN.rst` describing the question, method, comparison budget, results, and limitations. Include it from `index.rst` after the preceding stage.
3. Generate its tables and figures from its archived numerical data. Add references for newly discussed methods.
4. Remove the completed stage's prospective description from `roadmap.rst`. Update the abstract and discussion to reflect the evidence, including null results.
5. Run `make check`, `make article-html`, and `make article`; inspect the PDF's equations, tables, figure, and references.

The revised roadmap has nine stages. Stages 7 (world models) and 8 (LLM overflow) are distinct application branches; stage 9 extends the LLM reader. As stages are completed, update the status table and the coverage map as well as the abstract. Once the intended experimental scope is complete, replace its prospective roadmap with a synthesis and finalize authorship, venue format, and reproducibility materials. Preserve the distinction between completed experiments and remaining hypotheses throughout.
