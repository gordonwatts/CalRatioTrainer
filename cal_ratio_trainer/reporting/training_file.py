from pathlib import Path
from typing import List
import pandas as pd
from pydantic import BaseModel
from dataclasses import dataclass

from cal_ratio_trainer.common.fileio import load_dataset


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


def make_report_plots(input_files: List[plot_file], cache: Path):
    # Open the data for plotting.
    files = [
        file_info(data=load_dataset(f.input_file, cache), legend_name=f.legend_name)
        for f in input_files
    ]
    print(files)
