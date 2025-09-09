import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pylatex import (
    Document, Section, Subsection, Subsubsection,
    Tabular, NoEscape, Package, Command, Environment
)
from pylatex.utils import bold


# --- Définition des environnements custom ---
class Itemize(Environment):
    _latex_name = "itemize"

class Enumerate(Environment):
    _latex_name = "enumerate"

class LstListing(Environment):
    _latex_name = "lstlisting"


class LatexGenerator:
    def __init__(self, filename: str, doc_class: str = "article"):
        path = Path(filename)
        self.output_dir = path.parent if path.parent != Path("") else Path(".")
        self.stem = path.stem
        self.tex_file = self.output_dir / f"{self.stem}.tex"
        self.pdf_file = self.output_dir / f"{self.stem}.pdf"

        self.doc = Document(default_filepath=str(self.tex_file.with_suffix("")), documentclass=doc_class)

    # ---------- Preamble ----------
    def add_package(self, package: str, options: Optional[str] = None) -> None:
        self.doc.packages.append(Package(package, options=options))

    def add_custom_preamble(self, code: str) -> None:
        self.doc.preamble.append(NoEscape(code))

    # ---------- Sections ----------
    def add_section(self, title: str) -> None:
        self.doc.append(Section(title))

    def add_subsection(self, title: str) -> None:
        self.doc.append(Subsection(title))

    def add_subsubsection(self, title: str) -> None:
        self.doc.append(Subsubsection(title))

    # ---------- Text ----------
    def add_text(self, text: str, bold_text: bool = False) -> None:
        self.doc.append(bold(text) if bold_text else text)

    # ---------- Lists ----------
    def add_list(self, items: List[str], ordered: bool = False) -> None:
        env = Enumerate if ordered else Itemize
        with self.doc.create(env()):
            for item in items:
                self.doc.append(NoEscape(rf"\item {item}"))

    # ---------- Tables ----------
    def add_table(
        self, df: pd.DataFrame, caption: str = "",
        use_booktabs: bool = True, escape: bool = False
    ) -> None:
        table = Tabular("|" + "c|" * len(df.columns))
        table.add_hline()
        table.add_row(df.columns.tolist())
        table.add_hline()
        for _, row in df.iterrows():
            table.add_row(row.tolist(), escape=escape)
        table.add_hline()
        if caption:
            self.doc.append(NoEscape(rf"\begin{{center}}\textbf{{{caption}}}\end{{center}}"))
        self.doc.append(table)

    # ---------- Code blocks ----------
    def add_code_block(self, code: str, language: str = "python") -> None:
        self.add_package("listings")
        self.add_custom_preamble(
            rf"\lstset{{basicstyle=\ttfamily,language={language},breaklines=true}}"
        )
        with self.doc.create(LstListing()):
            self.doc.append(NoEscape(code))

    # ---------- Generic Environments ----------
    def add_environment(self, env_name: str, content: str) -> None:
        self.doc.append(NoEscape(rf"\begin{{{env_name}}} {content} \end{{{env_name}}}"))

    # ---------- Abstract ----------
    def add_abstract(self, content: str) -> None:
        self.add_environment("abstract", content)

    # ---------- Appendix ----------
    def add_appendix(self) -> None:
        self.doc.append(Command("appendix"))

    # ---------- Build ----------
    def generate_pdf(self) -> None:
        self.doc.generate_tex()

        commands = [
            ["latexmk", "-pdf", str(self.tex_file)],
            ["pdflatex", str(self.tex_file)],  # premier passage
            ["pdflatex", str(self.tex_file)],  # second passage pour refs
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True, cwd=self.output_dir)
                return
            except FileNotFoundError:
                continue
            except subprocess.CalledProcessError as e:
                print("Erreur LaTeX :", e.stderr.decode() if e.stderr else e)
                raise

        raise RuntimeError("Aucun compilateur LaTeX valide trouvé (latexmk ou pdflatex).")