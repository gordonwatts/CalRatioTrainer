from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import pandas as pd

from cal_ratio_trainer.common.fileio import load_dataset
from cal_ratio_trainer.config import ReportingConfig

from .md_report import MDReport


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


def make_comparison_plots(
    col_names: Union[str, List[str]],
    ds_generator: Iterator[Tuple[pd.DataFrame, str]],
    by_string: str,
):
    """Make a comparison plot for a given column name. The generator should yield
    tuples of (data, name) where data is a pandas dataframe and name is a string
    that will be used in the legend.

    Args:
        col_names (Union[str, List[str]]): Thames of the column of the DF we will plot
        ds_generator (Generator[Tuple[pd.DataFrame, str], None, None]): Generator
            that yields tuples of (data, name) where data is a pandas dataframe and
            name is a string that will be used in the legend.
        by_string (str): The string to use in the title of the plot.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    # If the column names is a string, then make it a list.
    if isinstance(col_names, str):
        col_names = [col_names]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    for data, name in ds_generator:
        ax.hist(
            data[find_column(col_names, data)],
            bins=100,
            histtype="step",
            label=name,
            density=True,
        )

    ax.legend()
    ax.set_xlabel(col_names[0])
    ax.set_ylabel("Number of Jets")
    ax.set_title(f"{col_names[0]} By {by_string}")

    return fig


def plot_comparison_for_plot_list(
    plots: Optional[List[Union[str, List[str]]]],
    ds_generator: Callable[[], Iterator[Tuple[pd.DataFrame, str]]],
    by_string: str,
):
    """Make a comparison plot for a given column name. The generator should yield
    tuples of (data, name) where data is a pandas dataframe and name is a string
    that will be used in the legend.

    Args:
        plots (Optional[List[Union[str, List[str]]]]): List of items to plot from each
                sequence of data frames.
        ds_generator (Callable[[], Iterator[Tuple[pd.DataFrame, str]]]): Generator that
                yields tuples of (data, name) where data is a pandas dataframe and name
                is a string that will be used in the legend.
        by_string (str): The string to use in the title of the plot.

    Yields:
        Generator[matplotlib.figure.Figure]: The figure containing the plot.
    """
    if plots is not None:
        for col_names in plots:
            yield make_comparison_plots(
                col_names,
                ds_generator(),
                by_string,
            )


def make_report_plots(cache: Path, config: ReportingConfig):
    # Open the data for plotting.
    assert config.input_files is not None
    files = [
        file_info(
            data=load_dataset(f.input_file, cache),
            legend_name=f.legend_name,
            source_name=f.input_file,
        )
        for f in config.input_files
    ]

    # Create the report.
    assert config.output_report is not None
    with MDReport(config.output_report, "CalRatio Training File Info") as report:
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
        for p in plot_comparison_for_plot_list(
            config.common_plots,
            lambda: ((f.data, f.legend_name) for f in files),
            "File",
        ):
            report.add_figure(p)
            plt.close(p)

        # See if we have any files that are "in common" for the signal or the
        # control. If so, do the same common plots.
        def plot_common_files(
            common_files: Iterable[file_info], common_labels: Optional[Dict[int, str]]
        ):
            if common_labels is not None:
                for k_dtype, v_dtype in common_labels.items():
                    report.header(f"### Comparisons for `{v_dtype}`")
                    for p in plot_comparison_for_plot_list(
                        config.common_plots,
                        lambda: (
                            (f.data[f.data["label"] == k_dtype], f.legend_name)
                            for f in common_files
                        ),
                        f"By File for {v_dtype}",
                    ):
                        report.add_figure(p)
                        plt.close(p)

        data_samples = [f for f in files if f.is_signal]
        if len(data_samples) > 1:
            report.header("## Training File Comparisons")
            plot_common_files(data_samples, config.data_labels_signal)

            # Next, lets do the same thing, but for the different signal samples.
            def to_tuples(
                mass_list: List[Dict[Hashable, int]]
            ) -> List[Tuple[int, int]]:
                return [(m["llp_mH"], m["llp_mS"]) for m in mass_list]

            all_masses = [
                f.data.groupby(["llp_mH", "llp_mS"])
                .size()
                .reset_index(name="count")
                .to_dict("records")
                for f in data_samples
            ]
            common_masses = set(to_tuples(all_masses[0])).intersection(
                *[to_tuples(a) for a in all_masses[1:]]
            )
            common_masses = sorted(common_masses)
            for c_m in common_masses:
                report.header(f"### Comparisons for $m_H={c_m[0]}$, $m_S={c_m[1]}$")
                for p in plot_comparison_for_plot_list(
                    config.common_plots,
                    lambda: (
                        (
                            f.data[
                                (f.data["llp_mH"] == c_m[0])
                                & (f.data["llp_mS"] == c_m[1])
                            ],
                            f.legend_name,
                        )
                        for f in data_samples
                    ),
                    f"By File for $m_H={c_m[0]}$, $m_S={c_m[1]}$",
                ):
                    report.add_figure(p)
                    plt.close(p)

        control_samples = [f for f in files if not f.is_signal]
        if len(control_samples) > 1:
            report.header("## Adversary File Comparisons")
            plot_common_files(control_samples, config.data_labels_control)
        report.write("")

        # Plot some info per file:
        report.header("Some file specific information:")
        for f in files:
            file_type = "control" if not f.is_signal else "signal"
            report.header(f"### {f.legend_name} ({file_type} file)")

            # Repeat the common plots for the different data labels
            labels = (
                config.data_labels_control
                if not f.is_signal
                else config.data_labels_signal
            )
            assert labels is not None

            for p in plot_comparison_for_plot_list(
                config.common_plots,
                lambda: (
                    (f.data[f.data["label"] == k], v)
                    for k, v in labels.items()  # type: ignore
                ),
                "Data Type",
            ):
                report.add_figure(p)
                plt.close(p)

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
        # Also make plots per column - this is quite expensive in time.
        report.header("## Column Information")

        column_names = set()
        for f in files:
            column_names.update(f.data.columns)

        def make_file_plots(
            col_name: str, files: List[Tuple[str, pd.DataFrame]]
        ) -> str:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)

            for name, data in files:
                ax.hist(data, bins=100, histtype="step", label=name, density=True)

            ax.legend()
            ax.set_xlabel(col_name)
            ax.set_ylabel("Number of Jets")
            ax.set_title(f"{col_name} By File")

            md_text = report.figure_md(fig, display_size=150)
            plt.close(fig)
            return md_text

        col_table_dict = [
            {
                "Column": c,
                "Plot": make_file_plots(
                    c,
                    [(f.legend_name, f.data[c]) for f in files if c in f.data.columns],
                )
                if config.plot_every_column
                else "",
                **{f.legend_name: "X" if c in f.data.columns else "" for f in files},
            }
            for c in sorted(column_names, key=lambda x: x.lower())
        ]
        heading_list = ["Column", *[f.legend_name for f in files]]
        if config.plot_every_column:
            heading_list.append("Plot")
        report.add_table(col_table_dict, heading_list)
