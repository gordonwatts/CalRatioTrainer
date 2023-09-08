from io import TextIOWrapper
from pathlib import Path
from typing import Dict, List, Optional


class MDReport:
    """Class to aid with writing a new MD Report.

    Usage:
    >>> with MDReport(Path("report.md"), "Report Title") as report:
    >>>     report.write("## Header")
    >>>     report.write("Some text")
    >>>     report.write("## Another Header")
    >>>     report.write("More text")
    """

    # Make sure we never overwrite a plot if
    # even if the name is the same!

    def __init__(self, path: Path, title: str):
        self.path = path
        self.file: Optional[TextIOWrapper] = None
        self.title = title
        self.plot_index: int = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w")
        self.file.write(f"# {self.title}\n\n")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.file is not None
        self.file.close()

    def write(self, text: str):
        assert self.file is not None
        self.file.write(text)
        self.file.write("\n")

    def header(self, text: str):
        "Add a header to the file. Include the ## in the text!"
        self.write(f"{text}\n")

    def add_table(
        self, data: List[Dict[str, str]], col_order: Optional[List[str]] = None
    ):
        """Add a table to the markdown.

        * Each row's data is a dictionary with the column name as the key.
        * The column order is given by `col_order`. If `col_order` is None,
          then the columns are sorted alphabetically.

        Args:
            data (List[Dict[str, str]]): Each entry in the list is  one row of data,
                                         keyed by column name.
            col_order (Optional[List[str]], optional): The list of columns, or all
                                            sorted alphabetically. Defaults to None.
        """
        # Build the list of columns
        if col_order is None:
            col_order = sorted(data[0].keys())

        # Write the header
        self.write("|" + "|".join(col_order) + "|")
        self.write("|" + "|".join(["---"] * len(col_order)) + "|")

        # Write the data
        for row in data:
            self.write("|" + "|".join([str(row[c]) for c in col_order]) + "|")

        self.write("")

    def add_figure(self, fig, display_size: int = 300):
        """Add a matplotlib figure to the reporting.

        We will:

        * Save the figure to a file in the reporting directory (png file)
        * Add a markdown image link to the file
        * Make the markdown clickable to display the figure full size.

        Args:
            fig (matplotlib Figure): The figure to be added.
        """
        self.write(self.figure_md(fig, display_size=display_size))

    def _save_figure(self, fig, display_size: int = 300) -> Path:
        """Save the figure as a mat plot lib and return the path to it

        * Save the figure to a file in the reporting directory (png file)

        Args:
            fig (matplotlib.figure.Figure): The figure to save to a local png file.

        Returns:
            Path: The written out png file for the figure.
        """
        # Save the figure to a file
        fig_name = f"{self.plot_index:04d}-{fig.get_axes()[0].get_title()}"
        self.plot_index += 1
        path = self.path.parent / f"{fig_name}.png"
        if path.exists():
            path.unlink()
        fig.savefig(path)

        return path

    def figure_md(self, fig, display_size: int = 300) -> str:
        """Save the figure as a mat plot lib and return the markdown that
        will reference it.

        * Save the figure to a file in the reporting directory (png file)
        * Make the markdown clickable to display the figure full size.

        Args:
            fig (matplotlib.figure.Figure): The figure to save to a local png file.
            display_size (int, optional): Width in pixels for the image. Defaults to
                300.

        Returns:
            str: The markdown that can be inserted into the markdown file to display
                this.
        """
        # Save the figure to a file
        path = self._save_figure(fig)

        # Add the markdown
        return f'<a href="{path.name}"><img src="{path.name}" width={display_size}></a>'

    def figure_link(self, fig, text: str) -> str:
        """Returns the markdown link to the figure with the given text.

        Args:
            fig (_type_): Figure to be saved to a png
            text (str): Text for the link

        Returns:
            str: The markdown that uses the text to refer to the figure.
        """
        # Save the figure
        path = self._save_figure(fig)

        # Return the file.
        return f"[{text}]({path.name})"
