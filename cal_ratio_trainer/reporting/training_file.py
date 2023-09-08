from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from cal_ratio_trainer.common.fileio import load_dataset
from cal_ratio_trainer.config import ReportingConfig

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


# Get the columns we want to plot for from each file:
def find_column(name: List[str], file: Union[file_info, pd.DataFrame]) -> str:
    d = file.data if isinstance(file, file_info) else file
    for n in name:
        if n in d.columns:
            return n
    raise ValueError(f"Asked to find column {name} in file but it is not there.")


def make_file_comparison_plot(files: List[file_info], col_names: Union[str, List[str]]):
    # If the column names is a string, then make it a list.
    if isinstance(col_names, str):
        col_names = [col_names]

    # Extract the data we will be plotting in common:
    data = [
        {"name": f.legend_name, "data": f.data[find_column(col_names, f)]}
        for f in files
    ]

    # Now, make a histogram of the data using matplotlib.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    for d in data:
        ax.hist(d["data"], bins=100, histtype="step", label=d["name"])

    ax.legend()
    ax.set_xlabel(col_names[0])
    ax.set_ylabel("Number of Jets")
    ax.set_title(f"{col_names[0]} by file")

    return fig


def make_label_comparison_plot(
    file: file_info, col_names: Union[str, List[str]], label_names: Dict[int, str]
):
    # If the column names is a string, then make it a list.
    if isinstance(col_names, str):
        col_names = [col_names]

    # Add a histogram for each label type:
    col_name = find_column(col_names, file)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    for label in label_names.keys():
        ax.hist(
            file.data[file.data["label"] == label][col_name],
            bins=100,
            histtype="step",
            label=f"{label_names[label]}",
        )

    ax.legend()
    ax.set_xlabel(col_name)
    ax.set_ylabel("Number of Jets")
    ax.set_title(f"{col_name} By Data Type")

    return fig


def make_comparison_plots(
    col_names: Union[str, List[str]],
    ds_generator: Generator[Tuple[pd.DataFrame, str], None, None],
    by_string: str,
):
    # If the column names is a string, then make it a list.
    if isinstance(col_names, str):
        col_names = [col_names]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    for data, name in ds_generator:
        ax.hist(
            data[find_column(col_names, data)], bins=100, histtype="step", label=name
        )

    ax.legend()
    ax.set_xlabel(col_names[0])
    ax.set_ylabel("Number of Jets")
    ax.set_title(f"{col_names[0]} By {by_string}")

    return fig


def plot_comparison_for_plot_list(
    plots: Optional[List[Union[str, List[str]]]],
    ds_generator: Callable[[], Generator[Tuple[pd.DataFrame, str], None, None]],
    by_string: str,
):
    if plots is not None:
        for col_names in plots:
            yield make_comparison_plots(
                col_names,
                ds_generator(),
                by_string,
            )


def make_report_plots(
    input_files: List[plot_file], cache: Path, output: Path, config: ReportingConfig
):
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

        # Plot the "headline" plots that are hopefully "in common".
        if config.common_plots is not None:
            report.header("## Common Plots")
            for col_names in config.common_plots:
                p = make_file_comparison_plot(files, col_names)
                report.add_figure(p)
                plt.close(p)

        # Next a list of all the columns:
        report.header("Some file specific information:")
        for f in files:
            file_type = "control" if not f.is_signal else "signal"
            report.header(f"### {f.legend_name} ({file_type} file)")

            # Repeat the common plots for the different data labels
            if config.common_plots is not None:
                labels = (
                    config.data_labels_control
                    if not f.is_signal
                    else config.data_labels_signal
                )
                assert labels is not None

                for col_names in config.common_plots:
                    p = make_label_comparison_plot(f, col_names, labels)
                    report.add_figure(p)
                    plt.close(p)
                report.write("")

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

                # And now the common plots by mass category
                plots = plot_comparison_for_plot_list(
                    config.common_plots,
                    lambda: (
                        (
                            f.data[
                                (f.data["llp_mH"] == d["llp_mH"])
                                & (f.data["llp_mS"] == d["llp_mS"])
                            ],
                            f"{d['llp_mH']}-{d['llp_mS']}",
                        )
                        for d in mass_counts
                    ),
                    "Mass Category",
                )
                for p in plots:
                    report.add_figure(p)
                    plt.close(p)

        # Dump the columns in each file as a table. The column names as rows, and a
        # table column for each file, with a check if that file has that particular
        # column.
        report.header("## Column Information")

        column_names = set()
        for f in files:
            column_names.update(f.data.columns)

        col_table_dict = [
            {
                "Column": c,
                **{f.legend_name: "X" if c in f.data.columns else "" for f in files},
            }
            for c in sorted(column_names, key=lambda x: x.lower())
        ]
        report.add_table(col_table_dict)
