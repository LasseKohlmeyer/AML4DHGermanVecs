import pandas as pd

import matplotlib.pyplot as plt
from typing import List


def plot_single_benchmark(df: pd.DataFrame, benchmark: str, grouping: str, coloring: str, paper_mode: bool = True):
    df_copy = df[[grouping, coloring, benchmark]]
    legend_labels = df_copy[coloring].drop_duplicates().iloc[::-1]
    df_copy = df_copy.set_index([grouping, coloring])
    if paper_mode:
        font_size = 14
        legend_size = 10
        plt.rc('font', family='serif', size=font_size)
        grid = False
        title = None

    else:
        font_size = 20
        legend_size = 12
        plt.rc('font', family='sans-serif', size=font_size)
        grid = True
        title = benchmark

    ax = df_copy.unstack().plot(kind="bar", title=title, grid=grid, alpha=0.85, rot=90)
    #     ax.spines['bottom'].set_position('zero')
    ax.set_xlabel(grouping, fontsize=font_size)
    ax.set_ylim(ymin=0)
    ax.set_ylabel("Score", fontsize=font_size)
    if len(legend_labels) > 1:
        ax.legend(legend_labels, title=coloring, title_fontsize=legend_size, fontsize=legend_size, loc="best")
    plt.show()
    # plot_single_benchmark(df, benchmark="HumanAssessment", grouping="Data set", stratification="Algorithm")
    # plot_single_benchmark(df, benchmark="HumanAssessment", grouping="Algorithm", stratification="Data set")


def plot_counts(df: pd.DataFrame, x_attribute: str, coloring: List[str], paper_mode: bool = True):
    columns = [x_attribute]
    columns.extend(coloring)
    legend_labels = coloring
    if paper_mode:
        font_size = 14
        legend_size = 10
        plt.rc('font', family='serif', size=font_size)
        grid = False
        title = None

    else:
        font_size = 20
        legend_size = 12
        plt.rc('font', family='sans-serif', size=font_size)
        grid = True
        title = "Concepts found"

    count_df = df[columns].drop_duplicates()
    ax = count_df.plot(x_attribute, coloring, kind="bar", alpha=0.75, title=title, grid=grid)
    ax.set_ylabel("Count score")
    if len(legend_labels) > 1:
        ax.legend(legend_labels, title="Counts", loc="best", title_fontsize=legend_size, fontsize=legend_size)
    plt.show()

    # plot_counts(df, x_attribute="Data set", stratification=["# Concepts"])


def mean_group_df(df: pd.DataFrame, group: str):
    return df.groupby(group).mean()
    # mean_group_df(df, group="Data set")








