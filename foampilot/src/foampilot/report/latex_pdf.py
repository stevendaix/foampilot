import pandas as pd
from pylatex import Document, Section, Subsection, Math, NoEscape, Figure, Package, MultiColumn, MultiRow, Tabular
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any

class LatexDocument:
    """A wrapper class for generating professional LaTeX documents and PDFs using PyLaTeX.

    This class simplifies the creation of LaTeX reports, handling the directory structure,
    document metadata, and common elements like tables, figures, and mathematical equations.

    Attributes:
        doc (Document): The PyLaTeX Document instance.
        filename (str): The name of the output file (without extension).
        parent_path (Path): The root directory where the report will be generated.
        report_path (Path): The specific subdirectory ('report/') for output files.
        filepath (Path): The full path to the generated file.
    """

    def __init__(
        self, 
        title: str, 
        author: str, 
        filename: str, 
        output_dir: Optional[Union[str, Path]] = None, 
        parent_path: Optional[Union[str, Path]] = None
    ):
        """Initializes the LatexDocument with metadata and directory setup.

        Args:
            title: The title of the document.
            author: The name of the author.
            filename: Name of the output file (e.g., 'simulation_report').
            output_dir: Directory where the 'report' folder will be created. 
                Defaults to the current working directory.
            parent_path: Alias for output_dir (for backward compatibility).
        """
        self.doc = Document()
        self.doc.preamble.append(NoEscape(r'\usepackage{graphicx}'))

        self.doc.preamble.append(NoEscape(r'\title{' + title + '}'))
        self.doc.preamble.append(NoEscape(r'\author{' + author + '}'))
        self.doc.preamble.append(NoEscape(r'\date{\today}'))

        self.filename = filename

        # Define parent directory
        if output_dir:
            self.parent_path = Path(output_dir)
        else:
            self.parent_path = Path.cwd()

        # Create report directory
        self.report_path = self.parent_path / "report"
        self.report_path.mkdir(parents=True, exist_ok=True)

        # Full path without extension
        self.filepath = self.report_path / self.filename

    def add_title(self):
        """Appends the `\maketitle` command to the document."""
        self.doc.append(NoEscape(r'\maketitle'))

    def add_toc(self):
        """Appends the table of contents command to the document."""
        self.doc.append(NoEscape(r'\tableofcontents'))

    def add_section(self, title: str, content: Union[str, NoEscape, Any]):
        """Adds a numbered section to the document.

        Args:
            title: The title of the section.
            content: The text or PyLaTeX objects to place inside the section.
        """
        with self.doc.create(Section(title)):
            self.doc.append(content)

    def add_subsection(self, title: str, content: Union[str, NoEscape, Any]):
        """Adds a numbered subsection.

        Args:
            title: The title of the subsection.
            content: The content to append.
        """
        with self.doc.create(Subsection(title)):
            self.doc.append(content)

    def add_unnumbered_section(self, title: str, content: str):
        """Adds a section that does not appear in the numbering or TOC.

        Args:
            title: Section title.
            content: Section text content.
        """
        self.doc.append(NoEscape(r'\section*{' + title + '}'))
        self.doc.append(content)

    def add_abstract(self, content: str):
        """Adds an abstract environment to the document.

        Args:
            content: The text for the abstract.
        """
        self.doc.append(NoEscape(r'\begin{abstract}'))
        self.doc.append(content)
        self.doc.append(NoEscape(r'\end{abstract}'))

    def add_list(self, items: List[str], ordered: bool = False):
        """Adds a bulleted (itemize) or numbered (enumerate) list.

        Args:
            items: A list of strings to be converted into list items.
            ordered: If True, uses 'enumerate'; otherwise uses 'itemize'.
        """
        list_type = 'enumerate' if ordered else 'itemize'
        self.doc.append(NoEscape(r'\begin{' + list_type + '}'))
        for item in items:
            self.doc.append(NoEscape(r'\item ' + item))
        self.doc.append(NoEscape(r'\end{' + list_type + '}'))

    def add_math(self, equation: str):
        """Adds a math equation using the standard LaTeX Math environment.

        Args:
            equation: LaTeX-formatted string (e.g., r'a^2 + b^2 = c^2').
        """
        self.doc.append(Math(data=[equation], escape=False))

    def add_figure(self, image_path: str, caption: Optional[str] = None, width: str = '0.8\\textwidth'):
        """Adds an image as a figure.

        Args:
            image_path: Path to the image file.
            caption: Optional text for the figure caption.
            width: String defining the width (e.g., r'0.5\textwidth').
        """
        with self.doc.create(Figure(position='h!')) as fig:
            fig.add_image(image_path, width=NoEscape(width))
            if caption:
                fig.add_caption(caption)

    def add_table(
        self, 
        data: List[List[Any]], 
        headers: Optional[List[str]] = None, 
        caption: str = "", 
        col_align: str = "c", 
        multicol_data: Optional[Dict[Tuple[int, int], Tuple[int, str, str]]] = None, 
        multirow_data: Optional[Dict[Tuple[int, int], Tuple[int, str]]] = None
    ):
        """Adds a LaTeX table with advanced cell merging capabilities.

        Args:
            data: A 2D list containing the table rows.
            headers: Optional list of column headers.
            caption: Optional caption. If provided, the table is wrapped in a Figure environment.
            col_align: Default alignment for columns ('c', 'l', or 'r').
            multicol_data: Dict mapping `(row, col)` to `(span, align, content)`.
            multirow_data: Dict mapping `(row, col)` to `(span, content)`.
        """
        num_cols = len(headers) if headers else len(data[0])
        col_format = f"|{'|'.join([col_align] * num_cols)}|"

        table = Tabular(col_format)
        table.add_hline()

        if headers:
            table.add_row(headers)
            table.add_hline()

        for row_idx, row in enumerate(data):
            formatted_row = []
            col_skip = 0
            row_skip = [False] * num_cols

            for col_idx, cell in enumerate(row):
                if row_skip[col_idx]: continue
                if col_skip > 0:
                    col_skip -= 1
                    continue

                if multicol_data and (row_idx, col_idx) in multicol_data:
                    span, align, content = multicol_data[(row_idx, col_idx)]
                    formatted_row.append(MultiColumn(span, align=align, data=content))
                    col_skip = span - 1
                elif multirow_data and (row_idx, col_idx) in multirow_data:
                    span, content = multirow_data[(row_idx, col_idx)]
                    formatted_row.append(MultiRow(span, data=content))
                    row_skip[row_idx + 1: row_idx + span] = [True] * (span - 1)
                else:
                    formatted_row.append(cell)

            while len(formatted_row) < num_cols:
                formatted_row.append("")

            table.add_row(formatted_row)
            table.add_hline()

        if caption:
            with self.doc.create(Figure(position='h!')) as table_env:
                table_env.append(table)
                table_env.add_caption(caption)
        else:
            self.doc.append(table)

    @classmethod
    def dataframe_to_latex(cls, dataframe: pd.DataFrame, caption: str = "", label: str = "") -> str:
        """Converts a pandas DataFrame into a raw LaTeX string.

        Args:
            dataframe: The DataFrame to convert.
            caption: Optional table caption.
            label: Optional label for cross-referencing.

        Returns:
            A string containing the LaTeX code for the table.
        """
        latex_table = dataframe.to_latex(index=False, escape=False)
        if caption:
            latex_table = f"\\begin{{table}}[h!]\n\\centering\n{latex_table}\n\\caption{{{caption}}}\n\\end{{table}}"
        if label:
            latex_table = latex_table.replace("\\begin{table}", f"\\begin{{table}}[h!]\n\\label{{{label}}}")
        return latex_table

    def add_dataframe_table(self, dataframe: pd.DataFrame, caption: str = "", label: str = ""):
        """Appends a pandas DataFrame as a table to the document.

        Args:
            dataframe: The DataFrame to add.
            caption: Optional caption.
            label: Optional label.
        """
        latex_table = self.dataframe_to_latex(dataframe, caption, label)
        self.doc.append(NoEscape(latex_table))

    def add_bibliography(self, bib_path: str):
        """Adds the bibliography section using BibTeX.

        Args:
            bib_path: Path to the .bib file.

        Raises:
            FileNotFoundError: If the .bib file is not found.
        """
        if not os.path.exists(bib_path):
            raise FileNotFoundError(f"Bibliography file not found: {bib_path}")
        self.doc.append(NoEscape(r'\bibliographystyle{plain}'))
        self.doc.append(NoEscape(r'\bibliography{' + bib_path + '}'))

    def add_environment(self, env_name: str, content: str):
        """Wraps content in a generic LaTeX environment.

        Args:
            env_name: Name of the environment (e.g., 'center', 'quote').
            content: Text to be placed inside the environment.
        """
        self.doc.append(NoEscape(r'\begin{' + env_name + '}'))
        self.doc.append(content)
        self.doc.append(NoEscape(r'\end{' + env_name + '}'))

    def add_appendix(self, title: str, content: str):
        """Initializes the appendix section and adds an appendix entry.

        Args:
            title: Title of the appendix section.
            content: Content text for the appendix.
        """
        self.doc.append(NoEscape(r'\appendix'))
        self.add_section(title, content)

    def add_package(self, package_name: str, options: Optional[Union[str, List[str]]] = None):
        """Adds a LaTeX package to the document preamble.

        Args:
            package_name: Name of the package to include.
            options: Optional package options.
        """
        if options:
            self.doc.packages.append(Package(package_name, options=options))
        else:
            self.doc.packages.append(Package(package_name))

    def add_custom_preamble(self, command: str):
        """Adds a raw LaTeX command to the document preamble.

        Args:
            command: The raw LaTeX command string.
        """
        self.doc.preamble.append(NoEscape(command))

    def generate_tex(self):
        """Generates the .tex source file in the report directory."""
        self.doc.generate_tex(str(self.filepath))

    def generate_pdf(self):
        """Compiles the .tex file into a PDF using pdflatex.

        Note:
            Runs pdflatex twice to ensure the Table of Contents (TOC) is updated.
        """
        subprocess.run(['pdflatex', f'{self.filename}.tex'], cwd=self.report_path)
        subprocess.run(['pdflatex', f'{self.filename}.tex'], cwd=self.report_path)

    def generate_document(self, output_format: str = "pdf"):
        """Main entry point to generate the output files.

        Args:
            output_format: Target format ('pdf' or 'tex'). 
                If 'pdf', the .tex file is generated first.
        """
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
