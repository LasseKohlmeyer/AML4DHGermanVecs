import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from typing import List


def plot_style(paper_mode: bool = True, title: str = None, horizontal: bool = False):
    if paper_mode:
        fig_size = None  # (8,8)
        font_size = 14
        legend_size = 10

        plt.rc('font', family='serif', size=font_size)
        grid = False
        title = None
        label_rot = 90
        x_al = "center"
    else:
        fig_size = (15, 10)
        font_size = 20
        legend_size = 12
        plt.rc('font', family='sans-serif', size=font_size)
        grid = True
        label_rot = 45
        x_al = "right"
    if horizontal:
        label_rot = 0

    return fig_size, font_size, legend_size, grid, title, label_rot, x_al


def legend(ax, paper_mode, legend_labels, title, legend_size, loc="best", horizontal: bool = False):
    def get_legend_label_from_str(input_string: str):
        return input_string.split(',')[-1].replace(')', '').strip()

    if len(legend_labels) > 1:
        if paper_mode:
            lgd = ax.legend(title=title, title_fontsize=legend_size, fontsize=legend_size, loc=loc)
            labels = [get_legend_label_from_str(te.get_text()) for te in lgd.get_texts()]
            lgd = ax.legend(labels, title=title, title_fontsize=legend_size, fontsize=legend_size, loc=loc)
        else:

            lgd = ax.legend(title=title, title_fontsize=legend_size, fontsize=legend_size, loc='center left',
                            bbox_to_anchor=(1.0, 0.5))
            labels = [get_legend_label_from_str(te.get_text()) for te in lgd.get_texts()]
            ax.legend(labels, title=title, title_fontsize=legend_size, fontsize=legend_size, loc='center left',
                      bbox_to_anchor=(1.0, 0.5))
    else:
        ax.get_legend().remove()


def axis_modification(ax, x_label, y_label, x_al, font_size, horizontal: bool = False):
    if horizontal:
        x_label_use = y_label
        y_label_use = x_label
    else:
        x_label_use = x_label
        y_label_use = y_label

    if x_label_use:
        ax.set_xlabel(x_label_use, fontsize=font_size)
    if y_label_use:
        ax.set_ylabel(y_label_use, fontsize=font_size)
    #     ax.set_ylim(ymin=0)
    if horizontal:
        ax.set_yticklabels(ax.get_yticklabels(), ha=x_al)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), ha=x_al)


def plot_single_benchmark(df: pd.DataFrame, benchmark: str, grouping: List[str], coloring: str, paper_mode: bool = True,
                          horizontal: bool = False):
    if coloring not in grouping:
        grouping = [*grouping, coloring]

    df_copy = df[[*grouping, benchmark]]
    legend_labels = df_copy[coloring].drop_duplicates().iloc[::-1]
    df_copy = df_copy.set_index(grouping)
    fig_size, font_size, legend_size, grid, title, label_rot, x_al = plot_style(paper_mode, title=benchmark,
                                                                                horizontal=horizontal)
    if horizontal:
        ax = df_copy.unstack().plot.barh(title=title, grid=grid, alpha=0.85, rot=label_rot, figsize=fig_size)
    else:
        ax = df_copy.unstack().plot(kind="bar", title=title, grid=grid, alpha=0.85, rot=label_rot, figsize=fig_size)
    #     ax.spines['bottom'].set_position('zero')

    #     ax.set_xlabel(grouping, fontsize=font_size)
    #     ax.set_ylim(ymin=0)
    #     ax.set_ylabel("Score", fontsize=font_size)
    #     ax.set_xticklabels(ax.get_xticklabels(),ha=x_al)
    axis_modification(ax, x_label=None, y_label=f"{benchmark} Score", x_al=x_al, font_size=font_size,
                      horizontal=horizontal)

    legend(ax=ax, paper_mode=paper_mode, legend_labels=legend_labels, title=coloring, legend_size=legend_size,
           horizontal=horizontal)

    plt.show()
    # plot_single_benchmark(df, benchmark="HumanAssessment", grouping="Data set", stratification="Algorithm")
    # plot_single_benchmark(df, benchmark="HumanAssessment", grouping="Algorithm", stratification="Data set")


def plot_counts(df: pd.DataFrame, x_attribute: str, grouping: str, coloring: List[str], paper_mode: bool = True):
    columns = [x_attribute]
    columns.extend(coloring)
    legend_labels = coloring

    fig_size, font_size, legend_size, grid, title, label_rot, x_al = plot_style(paper_mode, title="Concepts found")
    count_df = df[columns].drop_duplicates()
    ax = count_df.plot(x_attribute, coloring, kind="bar", alpha=0.75, title=title, grid=grid, rot=label_rot,
                       figsize=fig_size)
    #     ax.set_ylabel("Count score")
    #     ax.set_xticklabels(ax.get_xticklabels(),ha=x_al)
    axis_modification(ax, x_label=None, y_label="Count score", x_al=x_al, font_size=font_size)
    legend(ax=ax, paper_mode=paper_mode, legend_labels=legend_labels, title="Counts", legend_size=legend_size)
    plt.show()

    # plot_counts(df, x_attribute="Data set", stratification=["# Concepts"])


def mean_group_df(df: pd.DataFrame, group: str):
    return df.groupby(group).mean()
    # mean_group_df(df, group="Data set")


def mean_std(data):
    return data.mean(), data.std()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def get_avg_gain(df, column, attribute, scoring_column):
    data = df.loc[df[column] == attribute][scoring_column]
    return mean_confidence_interval(data)


def get_avg_gains(df, column, scoring_columns):
    attributes = set(df[column])
    tuples = []
    for attribute in attributes:
        for scoring_column in scoring_columns:
            m, std = get_avg_gain(df, column, attribute, scoring_column)
            #             print(f'{attribute}[{scoring_column}]: {m} [{std}])')
            tuples.append((attribute, scoring_column, m, std))

    df = pd.DataFrame.from_records(tuples)
    df.columns = ["Attribute", "Scoring", "Mean", "STD"]
    return df


def plot_avg_df(df: pd.DataFrame, paper_mode: bool = False, color_group_name=None, horizontal: bool = True):
    grouping = ["Scoring", "Attribute"]
    coloring = ["Attribute"]

    legend_labels = df[coloring].drop_duplicates().iloc[::-1]
    errors = df[["STD"]]
    df = df.set_index(grouping)
    fig_size, font_size, legend_size, grid, title, label_rot, x_al = plot_style(paper_mode,
                                                                                title=f"{color_group_name} Comparison",
                                                                                horizontal=horizontal)
    unstacked_df = df.unstack()
    if horizontal:
        ax = unstacked_df["Mean"].plot.barh(title=title, grid=grid, alpha=0.85, rot=label_rot, figsize=fig_size,
                                            xerr=unstacked_df["STD"])
    else:
        ax = unstacked_df["Mean"].plot(kind="bar", title=title, grid=grid, alpha=0.85, rot=label_rot, figsize=fig_size,
                                       yerr=unstacked_df["STD"])
    axis_modification(ax, x_label="Benchmark", y_label=f"Benchmark Score", x_al=x_al, font_size=font_size,
                      horizontal=horizontal)

    legend(ax=ax, paper_mode=paper_mode, legend_labels=legend_labels, title=color_group_name, legend_size=legend_size,
           horizontal=horizontal)

    plt.show()


def plot_comparison(df, comparison_attribute, benchmarks, papermode):
    comparison_df = get_avg_gains(df, comparison_attribute, benchmarks)
    plot_avg_df(comparison_df, color_group_name=comparison_attribute)


