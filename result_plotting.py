import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def plot_single_benchmark(df: pd.DataFrame, benchmark: str, grouping: str, stratification: str):
    df_copy = df[[grouping, stratification, benchmark]]
    legend_labels = df_copy[stratification].drop_duplicates()
    df_copy = df_copy.set_index([grouping, stratification])
    ax = df_copy.unstack().plot(kind="bar", title=benchmark)
    ax.set_ylabel("Score")
    ax.legend(legend_labels, title=stratification)
    plt.show()
    # plot_single_benchmark(df, benchmark="HumanAssessment", grouping="Data set", stratification="Algorithm")
    # plot_single_benchmark(df, benchmark="HumanAssessment", grouping="Algorithm", stratification="Data set")


def plot_counts(df: pd.DataFrame, x_attribute: str, stratification: List[str]):
    columns = [x_attribute]
    columns.extend(stratification)
    legend_labels = stratification
    count_df = df[columns].drop_duplicates()
    ax = count_df.plot(x_attribute, stratification, kind="bar", title="Concepts found")
    ax.set_ylabel("Count score")
    ax.legend(legend_labels, title="Counts")
    plt.show()

    # plot_counts(df, x_attribute="Data set", stratification=["# Concepts"])

    def mean_group_df(df: pd.DataFrame, group: str):
        return df.groupby(group).mean()
    # mean_group_df(df, group="Data set")



