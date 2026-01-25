import pandas as pd
import typst
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any

class TypstScientificReport:
    """
    Une classe complète pour générer des rapports scientifiques professionnels.
    Remplace avantageusement PyLaTeX par la puissance et la rapidité de Typst.
    """

    def __init__(
        self, 
        title: str, 
        author: str, 
        filename: str, 
        output_dir: Optional[Union[str, Path]] = None
    ):
        self.title = title
        self.author = author
        self.filename = filename
        self.parent_path = Path(output_dir) if output_dir else Path.cwd()
        self.report_path = self.parent_path / "report"
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.filepath = self.report_path / f"{self.filename}.pdf"

        self.content = []
        self._setup_document()

    def _setup_document(self):
        """Configuration du template scientifique (Preamble)."""
        setup = f"""
        #set document(title: "{self.title}", author: "{self.author}")
        #set page(paper: "a4", margin: 2.5cm, numbering: "1 / 1")
        #set text(font: "New Computer Modern", size: 11pt, lang: "fr")
        #set heading(numbering: "1.1.")
        #set par(justify: true)
        
        // Style pour les légendes
        #show figure.caption: it => [
          #text(weight: "bold", size: 0.9em)[#it.supplement #it.counter.display():] #it.body
        ]
        """
        self.content.append(setup)

    def add_title_page(self):
        """Crée une page de garde élégante."""
        title_page = f"""
        #align(center + horizon)[
            #block(text(weight: "bold", size: 2.5em)[{self.title}])
            #v(2em)
            #text(size: 1.5em)[{self.author}]
            #v(1em)
            #line(length: 60%, stroke: 1pt)
            #v(1em)
            #text(size: 1.1em)[#datetime.today().display("[day] [month repr:long] [year]")]
        ]
        #pagebreak()
        """
        self.content.append(title_page)

    def add_toc(self):
        self.content.append("== Table des matières\n#outline(title: none, indent: auto)\n#pagebreak()")

    def add_abstract(self, content: str):
        self.content.append(f'#align(center)[#block(width: 85%)[*Résumé* \n #v(0.5em) {content}]]\n#v(2em)')

    def add_section(self, title: str, content: str = "", label: Optional[str] = None):
        lbl = f" <{label}>" if label else ""
        self.content.append(f"= {title}{lbl}\n{content}")

    def add_subsection(self, title: str, content: str = "", label: Optional[str] = None):
        lbl = f" <{label}>" if label else ""
        self.content.append(f"== {title}{lbl}\n{content}")

    def add_math(self, equation: str, caption: Optional[str] = None, label: Optional[str] = None):
        """Ajoute une équation numérotée."""
        lbl = f" <{label}>" if label else ""
        eq = f"$ {equation} ${lbl}"
        if caption:
            self.content.append(f'#figure({eq}, caption: [{caption}])')
        else:
            self.content.append(f" {eq} ")

    def add_figure(self, image_path: str, caption: str, label: str, width: str = "80%"):
        """Ajoute une image avec label pour référence croisée."""
        fig = f'#figure(image("{image_path}", width: {width}), caption: [{caption}]) <{label}>'
        self.content.append(fig)

    def add_list(self, items: List[str], ordered: bool = False):
        marker = "+" if ordered else "-"
        list_str = "\n".join([f"{marker} {item}" for item in items])
        self.content.append(list_str)

    def add_table(
        self, 
        data: List[List[Any]], 
        headers: Optional[List[str]] = None, 
        caption: str = "", 
        label: str = "",
        merges: List[Dict] = None # [{'row':0, 'col':0, 'colspan':2, 'rowspan':1, 'content': 'Test'}]
    ):
        """Tableau avancé avec support des fusions (colspan/rowspan)."""
        num_cols = len(headers) if headers else len(data[0])
        
        table_code = f'#figure(table(columns: {num_cols}, align: center + horizon, stroke: 0.5pt, inset: 8pt,\n'
        
        # Headers
        if headers:
            table_code += "  table.header(" + ", ".join([f"[* {h} *]" for h in headers]) + "),\n"
        
        # Gestion des cellules fusionnées (Merges)
        used_cells = set()
        if merges:
            for m in merges:
                r, c = m['row'], m['col']
                cs, rs = m.get('colspan', 1), m.get('rowspan', 1)
                txt = m['content']
                table_code += f'  table.cell(colspan: {cs}, rowspan: {rs}, fill: gray.lighten(80%))[{txt}],\n'
                for i in range(r, r + rs):
                    for j in range(c, c + cs):
                        used_cells.add((i, j))

        # Remplissage du reste des données
        for r_idx, row in enumerate(data):
            for c_idx, cell in enumerate(row):
                if (r_idx, c_idx) not in used_cells:
                    table_code += f'  [{cell}],\n'
        
        table_code += f'), caption: [{caption}]) <{label}>'
        self.content.append(table_code)

    def add_reference(self, label: str):
        """Retourne le code pour citer une figure/table/section (ex: @fig1)."""
        return f" @{label} "

    def generate(self, open_pdf: bool = False):
        """Compile le rapport."""
        full_source = "\n\n".join(self.content)
        
        # Sauvegarde du source .typ (toujours utile)
        typ_file = self.report_path / f"{self.filename}.typ"
        typ_file.write_text(full_source, encoding="utf-8")
        
        # Compilation
        try:
            typst.compile(full_source, output=str(self.filepath))
            print(f"✅ Rapport généré avec succès : {self.filepath}")
        except Exception as e:
            print(f"❌ Erreur de compilation : {e}")

# --- EXEMPLE D'UTILISATION SCIENTIFIQUE ---
if __name__ == "__main__":
    report = TypstScientificReport(
        title="Analyse de la propagation des ondes en milieu complexe",
        author="Dr. Jean Dupont",
        filename="rapport_final"
    )

    report.add_title_page()
    report.add_toc()
    
    report.add_abstract("Ce rapport présente les résultats de simulation...")

    report.add_section("Introduction", "Comme nous le verrons dans la @resultats...")
    
    # Ajout d'une équation complexe
    report.add_math(r"nabla^2 phi - 1/c^2 (partial^2 phi)/(partial t^2) = 0", 
                    caption="Équation d'onde de d'Alembert", label="eq_onde")

    # Ajout d'un tableau avec fusion de cellules
    data = [
        ["Exp 1", "0.23", "0.45"],
        ["Exp 2", "0.25", "0.48"],
    ]
    # On veut une ligne de titre fusionnée en haut (en plus des headers)
    merges = [{'row': 0, 'col': 0, 'colspan': 3, 'content': 'Résultats de Laboratoire'}]
    
    report.add_table(data, headers=["Test", "V1", "V2"], 
                    caption="Mesures expérimentales", label="tab_mesures", 
                    merges=merges)

    report.add_section("Résultats", "Les données récoltées sont présentées dans la @tab_mesures.", label="resultats")
    
    # Compilation
    report.generate()
