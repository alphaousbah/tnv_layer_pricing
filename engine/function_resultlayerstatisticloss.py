import numpy as np
import pandas as pd
from sqlalchemy import select

from db import LayerYearLoss, engine

PERCENTILES = [
    0.999,
    0.998,
    0.996,
    0.995,
    0.99,
    0.98,
    0.9667,
    0.96,
    0.95,
    0.9,
    0.8,
    0.5,
]


def get_df_resultlayerstatisticloss(layer_id, simulated_years):
    df_layeryearloss = get_df_layeryearloss(layer_id)
    df_aep = get_df_aep(df_layeryearloss, simulated_years)
    df_oep = get_df_oep(df_layeryearloss, simulated_years)
    df = pd.concat([df_aep, df_oep])
    df["layer_id"] = layer_id
    return df


def get_df_aep(df_layeryearloss, simulated_years):
    # Initialize get_resultlayerstatisticloss_aep
    df = pd.DataFrame({"percentile": PERCENTILES})
    df["statistic"] = "AEP"

    df_by_year = df_layeryearloss[["year", "ceded"]].groupby(by="year").sum()

    no_years_without_ceded_loss = simulated_years - len(df_by_year)
    df_years_without_ceded_loss = pd.DataFrame(
        np.zeros(no_years_without_ceded_loss), columns=["ceded"]
    )
    df_by_year = pd.concat([df_by_year, df_years_without_ceded_loss])

    df["loss"] = df["percentile"].map(
        lambda x: int(df_by_year["ceded"].quantile(x, interpolation="higher"))
    )

    return df


def get_df_oep(df_layeryearloss, simulated_years):
    # Initialize get_resultlayerstatisticloss_aep
    df = pd.DataFrame({"percentile": PERCENTILES})
    df["statistic"] = "OEP"

    df_by_year = df_layeryearloss[["year", "ceded"]].groupby(by="year").max()

    no_years_without_ceded_loss = simulated_years - len(df_by_year)
    df_years_without_ceded_loss = pd.DataFrame(
        np.zeros(no_years_without_ceded_loss), columns=["ceded"]
    )
    df_by_year = pd.concat([df_by_year, df_years_without_ceded_loss])

    df["loss"] = df["percentile"].map(
        lambda x: int(df_by_year["ceded"].quantile(x, interpolation="higher"))
    )

    return df


def get_df_layeryearloss(layer_id):
    query = select(LayerYearLoss).filter_by(layer_id=layer_id)
    df = pd.read_sql_query(query, engine)
    return df
