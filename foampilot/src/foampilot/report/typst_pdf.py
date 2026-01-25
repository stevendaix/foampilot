from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import typst
import subprocess

# ============================================================
# Utils
# ============================================================

def typst_escape(text: str) -> str:
    """Échappement pour le texte brut afin d'éviter les erreurs de syntaxe Typst."""
    return (
        text.replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("$", "\\$")
    )

# ============================================================
# Modèles logiques
# ============================================================

class Section:
    def __init__(self, title: str, content: str, level: int, label: Optional[str]):
        self.title, self.content, self.level, self.label = title, content, level, label

class Equation:
    def __init__(self, formula: str, caption: Optional[str], label: Optional[str]):
        self.formula, self.caption, self.label = formula, caption, label

class Figure:
    def __init__(self, path: str, caption: str, label: str, width: str):
        self.path, self.caption, self.label, self.width = path, caption, label, width

class Table:
    def __init__(self, data: List[List[Any]], headers: Optional[List[str]], 
                 caption: str, label: Optional[str], merges: List[Dict]):
        self.data, self.headers, self.caption, self.label, self.merges = data, headers, caption, label, merges

class CodeBlock:
    def __init__(self, code: str, lang: str, caption: Optional[str]):
        self.code, self.lang, self.caption = code, lang, caption

class Bibliography:
    def __init__(self, path: str, style: str):
        self.path, self.style = path, style

# ============================================================
# Document scientifique
# ============================================================

class ScientificDocument:
    def __init__(self, title: str, author: str):
        self.title = title
        self.author = author
        self.elements: List[Union[Section, Equation, Figure, Table, CodeBlock]] = []
        self.bib: Optional[Bibliography] = None
        self._labels: set[str] = set()

    def _register_label(self, label: Optional[str]):
        if label:
            if label in self._labels:
                raise ValueError(f"Label dupliqué : {label}")
            self._labels.add(label)

    def ref(self, label: str) -> str:
        """Référence croisée vers une figure, table ou section (ex: @fig_1)."""
        return f"@{label}"

    def cite(self, key: str) -> str:
        """Citation bibliographique (ex: @article2024)."""
        return f"@{key}"

    def add_section(self, title: str, content: str = "", level: int = 1, label: Optional[str] = None):
        self._register_label(label)
        self.elements.append(Section(title, content, level, label))

    def add_equation(self, formula: str, caption: Optional[str] = None, label: Optional[str] = None):
        self._register_label(label)
        self.elements.append(Equation(formula, caption, label))

    def add_figure(self, path: str, caption: str, label: str, width: str = "80%"):
        self._register_label(label)
        self.elements.append(Figure(path, caption, label, width))

    def add_table(self, data: List[List[Any]], headers: Optional[List[str]] = None, 
                  caption: str = "", label: Optional[str] = None, merges: Optional[List[Dict]] = None):
        self._register_label(label)
        self.elements.append(Table(data, headers, caption, label, merges or []))

    def add_code(self, code: str, lang: str = "python", caption: Optional[str] = None):
        self.elements.append(CodeBlock(code, lang, caption))

    def set_bibliography(self, path: str, style: str = "ieee"):
        """Définit le fichier de biblio (.bib ou .yml)."""
        self.bib = Bibliography(path, style)

# ============================================================
# Renderer Typst
# ============================================================

class TypstRenderer:
    def render(self, doc: ScientificDocument) -> str:
        parts = [self._preamble(doc)]
        
        for elem in doc.elements:
            if isinstance(elem, Section): parts.append(self._render_section(elem))
            elif isinstance(elem, Equation): parts.append(self._render_equation(elem))
            elif isinstance(elem, Figure): parts.append(self._render_figure(elem))
            elif isinstance(elem, Table): parts.append(self._render_table(elem))
            elif isinstance(elem, CodeBlock): parts.append(self._render_code(elem))
        
        if doc.bib:
            parts.append(f'\n#bibliography("{doc.bib.path}", style: "{doc.bib.style}")')
        
        return "\n\n".join(parts)

    def _preamble(self, doc: ScientificDocument) -> str:
        return f"""
#set document(title: "{typst_escape(doc.title)}", author: "{typst_escape(doc.author)}")
#set page(paper: "a4", margin: 2.5cm, numbering: "1 / 1")
#set text(font: "New Computer Modern", size: 11pt, lang: "fr")
#set heading(numbering: "1.1.")
#set par(justify: true)
#show figure.caption: it => [
  #text(weight: "bold", size: 0.9em)[#it.supplement #it.counter.display():] #it.body
]
"""

    def _render_section(self, s: Section) -> str:
        prefix = "=" * s.level
        lbl = f" <{s.label}>" if s.label else ""
        return f"{prefix} {typst_escape(s.title)}{lbl}\n{s.content}"

    def _render_equation(self, e: Equation) -> str:
        content = f"$ {e.formula} $"
        # Figure avec caption
        if e.caption:
            # ajouter le label après la figure avec <>
            lbl = f" <{e.label}>" if e.label else ""
            return f'#figure({content}, caption: [{typst_escape(e.caption)}]){lbl}'
        else:
            # align sans figure
            lbl = f" <{e.label}>" if e.label else ""
            return f"#align(center)[{content}]{lbl}"
        
    def _render_figure(self, f: Figure) -> str:
        return f'#figure(image("{f.path}", width: {f.width}), caption: [{typst_escape(f.caption)}]) <{f.label}>'

    def _render_table(self, t: Table) -> str:
        cols = len(t.headers) if t.headers else len(t.data[0])
        lbl = f" <{t.label}>" if t.label else ""
        code = [f'#figure(table(columns: {cols}, stroke: 0.5pt, inset: 7pt, align: center + horizon,']
        if t.headers:
            code.append(f"  table.header({', '.join(f'[* {typst_escape(h)} *]' for h in t.headers)}),")
        
        used = set()
        for m in t.merges:
            code.append(f'  table.cell(colspan: {m.get("colspan",1)}, rowspan: {m.get("rowspan",1)})[{typst_escape(m["content"])}],')
            for r in range(m.get("rowspan", 1)):
                for c in range(m.get("colspan", 1)): used.add((m["row"] + r, m["col"] + c))

        for r, row in enumerate(t.data):
            for c, cell in enumerate(row):
                if (r, c) not in used: code.append(f"  [{typst_escape(str(cell))}],")
        
        code.append(f'), caption: [{typst_escape(t.caption)}]){lbl}')
        return "\n".join(code)

    def _render_code(self, c: CodeBlock) -> str:
        block = f'``` {c.lang}\n{c.code}\n```'
        return f'#figure({block}, caption: [{typst_escape(c.caption)}])' if c.caption else block


    def compile_pdf(self, doc: ScientificDocument, output_pdf: str = "report/rapport_complet.pdf"):
        """
        Génère un fichier .typ et compile le PDF via le binaire Typst (subprocess).
        Compatible Ubuntu + Snap.
        """
        Path("report").mkdir(exist_ok=True)
        typ_file = Path("report/rapport_complet.typ")
        # Rendu Typst
        source = self.render(doc)
        typ_file.write_text(source, encoding="utf-8")

        # Chemin vers le binaire Typst
        typst_bin = "/snap/bin/typst"  # Modifie si nécessaire

        # Compilation via subprocess
        try:
            subprocess.run(
                [typst_bin, "compile", str(typ_file), output_pdf],  # <-- sortie en second argument
                check=True
            )
            print(f"PDF généré avec succès : {output_pdf}")
        except subprocess.CalledProcessError as e:
            print("Erreur lors de la compilation Typst :", e)
        except FileNotFoundError:
            print(f"Binaire Typst introuvable. Vérifie le chemin : {typst_bin}")



# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================

if __name__ == "__main__":
    doc = ScientificDocument("Analyse Avancée", "Dr. Isaac Netton")
    
    # 1. Configuration Biblio (optionnel)
    # doc.set_bibliography("ma_base.bib", style="ieee")

    # 2. Ajout de contenu dans l'ordre
    doc.add_section("Introduction", f"Comme vu dans @eq_onde, le modèle est stable.")
    
    doc.add_equation(r"f(x) = sqrt(1 + x^2)", caption="Fonction de base", label="eq_onde")
    
    doc.add_section("Code Source", "Voici l'implémentation Python :")
    doc.add_code("def hello():\n    print('Hello Typst!')", lang="python", caption="Exemple de code")

    # 3. Rendu et Compilation
    renderer = TypstRenderer()
    source = renderer.render(doc)
    
    Path("report").mkdir(exist_ok=True)
    
    renderer.compile_pdf(doc, output_pdf="report/rapport_complet.pdf")
    
    print("PDF généré avec succès dans le dossier 'report/'")
