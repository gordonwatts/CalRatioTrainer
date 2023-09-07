from pathlib import Path
from typing import List
import pandas as pd
from pydantic import BaseModel
from dataclasses import dataclass

from cal_ratio_trainer.common.fileio import load_dataset
from .md_report import MDReport


class plot_file(BaseModel):
    "Information about a plot file"

    # The url that we can use to read this file
    input_file: str

    # The name to use on the legend of the plot for data from this file.
    legend_name: str


@dataclass
class file_info:
    data: pd.DataFrame
    legend_name: str
    source_name: str

    @property
    def is_signal(self):
        return "llp_mH" in self.data.columns


def make_report_plots(input_files: List[plot_file], cache: Path, output: Path):
    # Open the data for plotting.
    files = [
        file_info(
            data=load_dataset(f.input_file, cache),
            legend_name=f.legend_name,
            source_name=f.input_file,
        )
        for f in input_files
    ]

    # Create the report.
    with MDReport(output, "CalRatio Training File Info") as report:
        # Create a dictionary for each of the files
        # which contain:
        # - The number of jets (rows) in the file
        # - How many jets of each label type (0, 1, or 2)

        file_info_table = [
            {
                "name": f.legend_name,
                "file": f.source_name,
                "Jets": len(f.data),
                "Signal (0)": len(f.data[f.data["label"] == 0]),
                "Multijet (1)": len(f.data[f.data["label"] == 1]),
                "BIB (2)": len(f.data[f.data["label"] == 2]),
            }
            for f in files
        ]

        report.header("## Training File Basics")
        report.write("What each file name actually points to.")
        report.add_table(file_info_table, ["name", "file"])
        report.write("Basic contents of the files")
        report.add_table(
            file_info_table, ["name", "Jets", "Signal (0)", "Multijet (1)", "BIB (2)"]
        )

        # Next a list of all the columns:
        report.header("Some file specific information:")
        for f in files:
            file_type = "control" if not f.is_signal else "signal"
            report.header(f"### {f.legend_name} ({file_type} file)")

            # Total up the number of events for each mass category (llp_mH, llp_mS)
            # that is found in the file.
            if f.is_signal:
                mass_counts = (
                    f.data.groupby(["llp_mH", "llp_mS"])
                    .size()
                    .reset_index(name="count")
                    .to_dict("records")
                )
                mass_table = [
                    {
                        "$m_H$, $m_S$": f"{d['llp_mH']}, {d['llp_mS']}",
                        "Jets": d["count"],
                    }
                    for d in mass_counts
                ]
                report.add_table(mass_table)

            # dump a list of columns for each
            report.write(f"{len(f.data.columns)} columns:\n")
            # Indent the list in markdown
            report.write(f"    {', '.join(f.data.columns)}")
