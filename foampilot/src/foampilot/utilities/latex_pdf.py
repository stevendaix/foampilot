import pandas as pd
from pylatex import Document, Section, Subsection, Math, NoEscape, Figure, Package, MultiColumn, MultiRow, Tabular
import os
import subprocess


class LatexDocument:
    """
    High-level LaTeX document generator with advanced formatting features.

    This class provides methods to create and format LaTeX documents with support for:
    - Sections and subsections
    - Abstracts
    - Tables with multi-column/row support
    - Mathematical equations
    - Figures and images
    - Bibliographies
    - Custom environments and packages

    Attributes
    ----------
    doc : Document
        The underlying pylatex Document object.
    filename : str
        Base filename for output files (without extension).
    """

    def __init__(self, title: str, author: str, filename: str):
        """
        Initialize a new LaTeX document with basic metadata.

        Parameters
        ----------
        title : str
            Document title.
        author : str
            Author name(s).
        filename : str
            Base filename for output (without .tex extension).
        """
        self.doc = Document()
        self.doc.preamble.append(NoEscape(r'\usepackage{graphicx}'))
        self.doc.preamble.append(NoEscape(r'\title{' + title + '}'))
        self.doc.preamble.append(NoEscape(r'\author{' + author + '}'))
        self.doc.preamble.append(NoEscape(r'\date{\today}'))
        self.filename = filename

    def add_title(self) -> None:
        """Add the document title using `\maketitle`."""
        self.doc.append(NoEscape(r'\maketitle'))

    def add_toc(self) -> None:
        """Add a table of contents to the document."""
        self.doc.append(NoEscape(r'\tableofcontents'))

    def add_section(self, title: str, content: str) -> None:
        """
        Add a numbered section.

        Parameters
        ----------
        title : str
            Section title.
        content : str
            Section content (can include LaTeX formatting).
        """
        with self.doc.create(Section(title)):
            self.doc.append(content)

    def add_subsection(self, title: str, content: str) -> None:
        """
        Add a numbered subsection.

        Parameters
        ----------
        title : str
            Subsection title.
        content : str
            Subsection content.
        """
        with self.doc.create(Subsection(title)):
            self.doc.append(content)

    def add_unnumbered_section(self, title: str, content: str) -> None:
        """
        Add an unnumbered section.

        Parameters
        ----------
        title : str
            Section title.
        content : str
            Section content.
        """
        self.doc.append(NoEscape(r'\section*{' + title + '}'))
        self.doc.append(content)

    def add_abstract(self, content: str) -> None:
        """
        Add an abstract section.

        Parameters
        ----------
        content : str
            Abstract text.
        """
        self.doc.append(NoEscape(r'\begin{abstract}'))
        self.doc.append(content)
        self.doc.append(NoEscape(r'\end{abstract}'))

    def add_list(self, items: list, ordered: bool = False) -> None:
        """
        Add a bulleted or numbered list.

        Parameters
        ----------
        items : list
            List of items to display.
        ordered : bool, optional
            If True, creates a numbered list. Default is False (bulleted list).
        """
        list_type = 'enumerate' if ordered else 'itemize'
        self.doc.append(NoEscape(r'\begin{' + list_type + '}'))
        for item in items:
            self.doc.append(NoEscape(r'\item ' + item))
        self.doc.append(NoEscape(r'\end{' + list_type + '}'))

    def add_math(self, equation: str) -> None:
        """
        Add a mathematical equation.

        Parameters
        ----------
        equation : str
            LaTeX-formatted equation string.
        """
        self.doc.append(Math(data=[equation], escape=False))

    def add_figure(self, image_path: str, caption: str = None, width: str = '0.8\\textwidth') -> None:
        """
        Add a figure with optional caption and width control.

        Parameters
        ----------
        image_path : str
            Path to image file.
        caption : str, optional
            Figure caption text.
        width : str, optional
            Image width as LaTeX dimension. Default is 0.8\textwidth.
        """
        with self.doc.create(Figure(position='h!')) as fig:
            fig.add_image(image_path, width=NoEscape(width))
            if caption:
                fig.add_caption(caption)

    def add_table(self, data: list, headers: list = None, caption: str = "",
                  col_align: str = "c", multicol_data: dict = None, multirow_data: dict = None) -> None:
        """
        Add a complex table with support for merged cells.

        Parameters
        ----------
        data : list of list
            2D list of table data.
        headers : list, optional
            Column headers.
        caption : str, optional
            Table caption.
        col_align : str, optional
            Column alignment ('c' for center, 'l' for left, 'r' for right). Default is 'c'.
        multicol_data : dict, optional
            Dictionary {(row,col): (span,align,content)} for column spans.
        multirow_data : dict, optional
            Dictionary {(row,col): (span,content)} for row spans.
        """
        num_cols = len(headers) if headers else len(data[0])
        col_format = "|".join([col_align] * num_cols)
        col_format = f"|{col_format}|"

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
                if row_skip[col_idx]:
                    continue
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
        """
        Convert pandas DataFrame to LaTeX table string.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame to convert.
        caption : str, optional
            Table caption.
        label : str, optional
            LaTeX label for referencing.

        Returns
        -------
        str
            LaTeX table code.
        """
        latex_table = dataframe.to_latex(index=False, escape=False)
        if caption:
            latex_table = f"\\begin{{table}}[h!]\n\\centering\n{latex_table}\n\\caption{{{caption}}}\n\\end{{table}}"
        if label:
            latex_table = latex_table.replace("\\begin{table}", f"\\begin{{table}}[h!]\n\\label{{{label}}}")
        return latex_table

    def add_dataframe_table(self, dataframe: pd.DataFrame, caption: str = "", label: str = "") -> None:
        """
        Add pandas DataFrame as table to the document.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame to add.
        caption : str, optional
            Table caption.
        label : str, optional
            LaTeX label.
        """
        latex_table = self.dataframe_to_latex(dataframe, caption, label)
        self.doc.append(NoEscape(latex_table))

    def add_bibliography(self, bib_path: str) -> None:
        """
        Add bibliography section.

        Parameters
        ----------
        bib_path : str
            Path to .bib file.

        Raises
        ------
        FileNotFoundError
            If the bibliography file does not exist.
        """
        if not os.path.exists(bib_path):
            raise FileNotFoundError(f"Bibliography file not found: {bib_path}")
        self.doc.append(NoEscape(r'\bibliographystyle{plain}'))
        self.doc.append(NoEscape(r'\bibliography{' + bib_path + '}'))

    def add_environment(self, env_name: str, content: str) -> None:
        """
        Add a custom LaTeX environment.

        Parameters
        ----------
        env_name : str
            Name of the LaTeX environment.
        content : str
            Content inside the environment.
        """
        self.doc.append(NoEscape(r'\begin{' + env_name + '}'))
        self.doc.append(content)
        self.doc.append(NoEscape(r'\end{' + env_name + '}'))

    def add_appendix(self, title: str, content: str) -> None:
        """
        Add an appendix section.

        Parameters
        ----------
        title : str
            Appendix title.
        content : str
            Appendix content.
        """
        self.doc.append(NoEscape(r'\appendix'))
        self.add_section(title, content)

    def add_package(self, package_name: str, options: str = None) -> None:
        """
        Add LaTeX package to preamble.

        Parameters
        ----------
        package_name : str
            Name of the package.
        options : str, optional
            Package options.
        """
        if options:
            self.doc.packages.append(Package(package_name, options=options))
        else:
            self.doc.packages.append(Package(package_name))

    def add_custom_preamble(self, command: str) -> None:
        """
        Add a custom LaTeX command to the preamble.

        Parameters
        ----------
        command : str
            LaTeX command string.
        """
        self.doc.preamble.append(NoEscape(command))

    def generate_tex(self) -> None:
        """Generate the .tex source file."""
        self.doc.generate_tex(self.filename)

    def generate_pdf(self) -> None:
        """Generate PDF by compiling LaTeX source (runs pdflatex twice)."""
        subprocess.run(['pdflatex', f'{self.filename}.tex'])
        subprocess.run(['pdflatex', f'{self.filename}.tex'])

    def generate_document(self, output_format: str = "pdf") -> None:
        """
        Generate final document in the specified format.

        Parameters
        ----------
        output_format : str, optional
            'tex' for LaTeX source or 'pdf' for compiled PDF. Default is 'pdf'.
        """
        self.generate_tex()
        if output_format == "pdf":
            self.generate_pdf()