import pandas as pd

import matplotlib.pyplot as plt
from typing import List

def plot_style(paper_mode: bool = True, title: str = None):
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

    return fig_size, font_size, legend_size, grid, title, label_rot, x_al


def legend(ax, paper_mode, legend_labels, title, legend_size, loc="best"):
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


def axis_modification(ax, x_label, y_label, x_al, font_size):
    if x_label:
        ax.set_xlabel(x_label, fontsize=font_size)
    if y_label:
        ax.set_ylabel(y_label, fontsize=font_size)
    #     ax.set_ylim(ymin=0)
    ax.set_xticklabels(ax.get_xticklabels(), ha=x_al)


def plot_single_benchmark(df: pd.DataFrame, benchmark: str, grouping: List[str], coloring: str,
                          paper_mode: bool = True):
    if coloring not in grouping:
        grouping = [*grouping, coloring]

    df_copy = df[[*grouping, benchmark]]

    legend_labels = df_copy[coloring].drop_duplicates().iloc[::-1]
    df_copy = df_copy.set_index(grouping)

    fig_size, font_size, legend_size, grid, title, label_rot, x_al = plot_style(paper_mode, title=benchmark)
    ax = df_copy.unstack().plot(kind="bar", title=title, grid=grid, alpha=0.85, rot=label_rot, figsize=fig_size)
    #     ax.spines['bottom'].set_position('zero')

    #     ax.set_xlabel(grouping, fontsize=font_size)
    #     ax.set_ylim(ymin=0)
    #     ax.set_ylabel("Score", fontsize=font_size)
    #     ax.set_xticklabels(ax.get_xticklabels(),ha=x_al)
    axis_modification(ax, x_label=None, y_label=f"{benchmark} Score", x_al=x_al, font_size=font_size)

    legend(ax=ax, paper_mode=paper_mode, legend_labels=legend_labels, title=coloring, legend_size=legend_size)

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



