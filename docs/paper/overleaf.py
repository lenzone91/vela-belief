"""Package the generated LaTeX sources for import into Overleaf."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def package_overleaf(app, exception):
    if exception is not None or app.builder.name != "latex":
        return

    output = Path(app.outdir)
    main = "vela-belief.tex"
    archive = output.parent / "vela-belief-overleaf.zip"
    # Include source dependencies only, even after a local PDF compilation.
    source_extensions = {".tex", ".sty", ".cls", ".xdy", ".ist", ".bib", ".bst"}
    figures = {Path(name) for name in app.builder.images.values()}
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if path.is_file() and (
                path.suffix in source_extensions
                or relative in figures
                or relative.as_posix() == "latexmkrc"
            ):
                bundle.write(path, relative.as_posix())
        bundle.writestr(
            "README.txt",
            "Import this ZIP using New Project > Upload Project in Overleaf.\n"
            f"Set Main document to {main} and Compiler to XeLaTeX.\n"
            "The project includes its Sphinx styles and generated figures.\n"
            "No Python, Sphinx, or Tectonic installation is needed on Overleaf.\n"
            "Permanent edits belong in docs/paper/ in the source repository;\n"
            "rebuilding replaces the generated LaTeX.\n",
        )
