import pandas as pd
from pylatex import Document, Section, Subsection, Math, NoEscape, Figure, Package, MultiColumn, MultiRow, Tabular
import os
import subprocess
from pylatex import Tabular, MultiColumn, MultiRow, Figure
from pathlib import Path

class LatexDocument:
    def __init__(self, title, author, filename, output_dir=None, parent_path=None):
        self.doc = Document()
        self.doc.preamble.append(NoEscape(r'\usepackage{graphicx}'))

        self.doc.preamble.append(NoEscape(r'\title{' + title + '}'))
        self.doc.preamble.append(NoEscape(r'\author{' + author + '}'))
        self.doc.preamble.append(NoEscape(r'\date{\today}'))

        self.filename = filename

        # Définir le dossier parent pour l'écriture
        if output_dir:
            self.parent_path = Path(output_dir)
        else:
            self.parent_path = Path.cwd()  # Par défaut le dossier courant

        # Créer le dossier report dans le dossier parent
        self.report_path = self.parent_path / "report"
        self.report_path.mkdir(parents=True, exist_ok=True)

        # Chemin complet du fichier sans extension
        self.filepath = self.report_path / self.filename

    def add_title(self):
        self.doc.append(NoEscape(r'\maketitle'))

    def add_toc(self):
        self.doc.append(NoEscape(r'\tableofcontents'))

    def add_section(self, title, content):
        with self.doc.create(Section(title)):
            self.doc.append(content)

    def add_subsection(self, title, content):
        with self.doc.create(Subsection(title)):
            self.doc.append(content)

    def add_unnumbered_section(self, title, content):
        self.doc.append(NoEscape(r'\section*{' + title + '}'))
        self.doc.append(content)

    def add_abstract(self, content):
        self.doc.append(NoEscape(r'\begin{abstract}'))
        self.doc.append(content)
        self.doc.append(NoEscape(r'\end{abstract}'))

    def add_list(self, items, ordered=False):
        list_type = 'enumerate' if ordered else 'itemize'
        self.doc.append(NoEscape(r'\begin{' + list_type + '}'))
        for item in items:
            self.doc.append(NoEscape(r'\item ' + item))
        self.doc.append(NoEscape(r'\end{' + list_type + '}'))

    def add_math(self, equation):
        """
        Adds a math equation to the document using the Math environment.

        :param equation: LaTeX-formatted string representing the equation.
        """
        self.doc.append(Math(data=[equation], escape=False))

    def add_figure(self, image_path, caption=None, width='0.8\\textwidth'):
        """
        Adds an image to the document with an optional caption and width.

        :param image_path: Path to the image file.
        :param caption: Optional caption for the image.
        :param width: Width of the image (default: 0.8\textwidth).
        """
        with self.doc.create(Figure(position='h!')) as fig:
            fig.add_image(image_path, width=NoEscape(width))
            if caption:
                fig.add_caption(caption)



    def add_table(self, data, headers=None, caption="", col_align="c", multicol_data=None, multirow_data=None):
        """
        Adds a table with support for MultiColumn and MultiRow.
        """
        # Calculate the number of columns based on the headers or the first row of data
        num_cols = len(headers) if headers else len(data[0])

        # Initialize the column format for the table (all centered by default)
        col_format = "|".join([col_align] * num_cols)
        col_format = f"|{col_format}|"

        # Create the Tabular object
        table = Tabular(col_format)
        table.add_hline()

        # Add headers if provided
        if headers:
            table.add_row(headers)
            table.add_hline()

        # Loop through each row in the data
        for row_idx, row in enumerate(data):
            formatted_row = []
            col_skip = 0  # Tracks columns skipped due to MultiColumn
            row_skip = [False] * num_cols  # To handle rows skipped by MultiRow

            for col_idx, cell in enumerate(row):
                if row_skip[col_idx]:
                    continue  # Skip the cell if part of a multirow

                if col_skip > 0:
                    col_skip -= 1
                    continue  # Skip this column if part of a multi-column

                if multicol_data and (row_idx, col_idx) in multicol_data:
                    # MultiColumn cell
                    span, align, content = multicol_data[(row_idx, col_idx)]
                    formatted_row.append(MultiColumn(span, align=align, data=content))
                    col_skip = span - 1  # Skip the columns that are merged
                elif multirow_data and (row_idx, col_idx) in multirow_data:
                    # MultiRow cell
                    span, content = multirow_data[(row_idx, col_idx)]
                    formatted_row.append(MultiRow(span, data=content))
                    row_skip[row_idx + 1: row_idx + span] = [True] * (span - 1)  # Skip the merged rows
                else:
                    # Regular cell
                    formatted_row.append(cell)

            # Adjust row length if MultiColumn was used
            while len(formatted_row) < num_cols:
                formatted_row.append("")  # Add empty cells to match the number of columns

            table.add_row(formatted_row)
            table.add_hline()

        # Wrap Tabular in a table environment if a caption is provided
        if caption:
            with self.doc.create(Figure(position='h!')) as table_env:
                table_env.append(table)
                table_env.add_caption(caption)
        else:
            # Append only the Tabular object
            self.doc.append(table)


    @classmethod
    def dataframe_to_latex(cls, dataframe: pd.DataFrame, caption: str = "", label: str = "") -> str:
        """
        Converts a pandas DataFrame into a LaTeX tabular string.

        :param dataframe: The pandas DataFrame to convert.
        :param caption: Caption for the table.
        :param label: Label for referencing the table.
        :return: A LaTeX table string.
        """
        latex_table = dataframe.to_latex(index=False, escape=False)
        if caption:
            latex_table = f"\\begin{{table}}[h!]\n\\centering\n{latex_table}\n\\caption{{{caption}}}\n\\end{{table}}"
        if label:
            latex_table = latex_table.replace("\\begin{table}", f"\\begin{{table}}[h!]\n\\label{{{label}}}")
        return latex_table

    def add_dataframe_table(self, dataframe: pd.DataFrame, caption: str = "", label: str = ""):
        """
        Adds a pandas DataFrame as a LaTeX table to the document.

        :param dataframe: The pandas DataFrame to add.
        :param caption: Caption for the table.
        :param label: Label for referencing the table.
        """
        latex_table = self.dataframe_to_latex(dataframe, caption, label)
        self.doc.append(NoEscape(latex_table))

    def add_bibliography(self, bib_path):
        if not os.path.exists(bib_path):
            raise FileNotFoundError(f"Bibliography file not found: {bib_path}")
        self.doc.append(NoEscape(r'\bibliographystyle{plain}'))
        self.doc.append(NoEscape(r'\bibliography{' + bib_path + '}'))

    def add_environment(self, env_name, content):
        self.doc.append(NoEscape(r'\begin{' + env_name + '}'))
        self.doc.append(content)
        self.doc.append(NoEscape(r'\end{' + env_name + '}'))

    def add_appendix(self, title, content):
        self.doc.append(NoEscape(r'\appendix'))
        self.add_section(title, content)

    def add_package(self, package_name, options=None):
        if options:
            self.doc.packages.append(Package(package_name, options=options))
        else:
            self.doc.packages.append(Package(package_name))

    def add_custom_preamble(self, command):
        self.doc.preamble.append(NoEscape(command))


    def generate_tex(self):
        self.doc.generate_tex(str(self.filepath))

    def generate_pdf(self):
        # Compile deux fois pour la TOC
        subprocess.run(['pdflatex', f'{self.filename}.tex'], cwd=self.report_path)
        subprocess.run(['pdflatex', f'{self.filename}.tex'], cwd=self.report_path)

    def generate_document(self, output_format="pdf"):
        self.generate_tex()
        if output_format == "pdf":
            self.generate_pdf()

# Example Usage
if __name__ == "__main__":
    doc = LatexDocument(title="Sample Document", author="Author Name", filename="sample_document")

    doc.add_title()
    doc.add_toc()
    doc.add_abstract("This is a sample abstract for the document.")
    doc.add_section("Introduction", "This is the introduction section.")
    doc.add_list(["Item 1", "Item 2", "Item 3"], ordered=True)
    doc.add_math(r"E = mc^2")
    doc.add_figure("example_image.png", caption="Exemple d'image", width="0.5\\textwidth")
    data = [
        ["Row 1, Col 1", "Row 1, Col 2", "Row 1, Col 3", "Row 1, Col 4"],
        ["Row 2, Col 1", "Row 2, Col 2", "Row 2, Col 3", "Row 2, Col 4"],
        ["Row 3, Col 1", "Row 3, Col 2", "Row 3, Col 3", "Row 3, Col 4"],
    ]

    # Spécification des fusions de colonnes
    multicol_data = {
        (0, 0): (2, "c", "Multicolumn spanning 2 columns"),  # Fusionne deux colonnes dans la première ligne
        (0, 2): (2, "c", "Multicolumn spanning 2 columns"),  # Fusionne deux colonnes dans la première ligne
    }

    # Spécification des fusions de lignes
    multirow_data = {
        (1, 0): (2, "MultiRow spanning 2 rows"),  # Fusionne deux lignes dans la première colonne
    }

    # En-têtes de colonnes
    headers = ["Header 1", "Header 2", "Header 3", "Header 4"]

    # Exemple d'appel de la fonction dans un document
    doc.add_table(
        data,
        headers=headers,
        caption="Table with MultiColumn and MultiRow"
    )


    #doc.add_bibliography("references.bib")
    doc.add_appendix("Appendix A", "This is appendix content.")

    # Generate the PDF
    doc.generate_document(output_format="pdf")
