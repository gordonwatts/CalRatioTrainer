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

    def __init__(self, path: Path, title: str):
        self.path = path
        self.file: Optional[TextIOWrapper] = None
        self.title = title

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
