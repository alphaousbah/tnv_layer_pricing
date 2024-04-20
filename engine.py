from time import perf_counter

import numpy as np
import pandas as pd
from numba import njit
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import Layer, LayerReinstatement, ModelYearLoss, engine

SIMULATED_YEARS = 2  # TODO: Change to 10000
pd.set_option("display.max_columns", None)
pd.options.mode.copy_on_write = True


def get_df_layeryearloss(df_layer_modelfile_table):
    df = pd.DataFrame([])  # Initialize df_layeryearlosses

    layer_ids = sorted(df_layer_modelfile_table["layer_id"].unique())
    for layer_id in layer_ids:
        linked_modelfiles_ids = sorted(
            df_layer_modelfile_table[df_layer_modelfile_table["layer_id"] == layer_id][
                "modelfile_id"
            ].astype(int)
        )
        df_layer = get_df_layeryearloss_single_layer(layer_id, linked_modelfiles_ids)

    return df


def get_df_layeryearloss_single_layer(layer_id, modelfiles_ids):
    start = perf_counter()
    layer = get_layer(layer_id)
    df = pd.DataFrame([])  # Initialize df_layeryearlosses_single_layer

    for modelfile_id in modelfiles_ids:
        df_modelyearlosses = get_df_modelyearlosses(modelfile_id)
        df = pd.concat([df, df_modelyearlosses], ignore_index=True)

    df = df.sort_values(["year", "day"])

    # Process recoveries
    (
        df["occ_recov_before_agg_deduct"],
        df["agg_deduct_before_occ"],
        df["occ_recov_after_agg_deduct"],
        df["agg_deduct_after_occ"],
        df["agg_limit_before_occ"],
        df["ceded"],
        df["cumulative_ceded"],
        df["agg_limit_after_occ"],
        df["net"],
    ) = get_occ_recoveries(
        df["year"].to_numpy(),
        df["gross"].to_numpy(),
        layer.occ_limit,
        layer.occ_deduct,
        layer.agg_limit,
        layer.agg_deduct,
    )

    # Process reinstatements
    df_by_year = df[["year", "gross", "ceded", "net"]].groupby(by="year").sum()
    expected_annual_loss = df_by_year["ceded"].sum() / SIMULATED_YEARS
    print(f"{expected_annual_loss=}")

    df_reinst = get_df_reinst(layer_id)
    if not df_reinst.empty:
        (
            df_reinst["limit_before_agg_limit"],
            df_reinst["deduct"],
            df_reinst["limit_after_agg_limit"],
        ) = get_reinst_limits(
            df_reinst["number"].to_numpy(), layer.agg_limit, layer.occ_limit
        )

        df_by_year["reinst_factor"] = get_reinst_factors(
            df_by_year["ceded"].to_numpy(),
            layer.occ_limit,
            df_reinst["rate"].to_numpy(),
            df_reinst["deduct"].to_numpy(),
            df_reinst["limit_after_agg_limit"].to_numpy(),
        )
        paid_premium = expected_annual_loss / (df_by_year["reinst_factor"].sum() / SIMULATED_YEARS)
        print(f"{paid_premium=}")

        (
            df["reinstated"],
            df["reinst_premium"],
        ) = get_occ_reinstatements(
            df["year"].to_numpy(),
            df["cumulative_ceded"].to_numpy(),
            layer.occ_limit,
            df_reinst["rate"].to_numpy(),
            df_reinst["deduct"].to_numpy(),
            df_reinst["limit_after_agg_limit"].to_numpy(),
            paid_premium,
        )

    else:
        df["reinstated"] = 0
        df["reinst_premium"] = 0

    end = perf_counter()
    print(f"Elapsed time: {end - start}")
    print(df)
    return df


def get_layer(layer_id):
    return Session(engine).get(Layer, int(layer_id))


def get_df_modelyearlosses(modelfile_id):
    query = select(ModelYearLoss).filter_by(modelfile_id=modelfile_id)
    df = pd.read_sql_query(query, engine)
    df = df.drop(columns="id")
    df = df.rename(columns={"loss": "gross"})
    return df


def get_df_reinst(layer_id):
    query = select(LayerReinstatement).filter_by(layer_id=layer_id)
    df = pd.read_sql_query(query, engine)
    df = df.drop(columns="id")
    df = df.drop(columns=["layer_id"])
    df = df.sort_values("order")
    return df


# Enhance loop performance using Numba's JIT
# https://pandas.pydata.org/docs/user_guide/enhancingperf.html#numba-jit-compilation
# See the Custom Function Example section
@njit
def get_occ_recoveries(
    year,
    gross,
    occ_limit,
    occ_deduct,
    agg_limit,
    agg_deduct,
):
    n = len(gross)  # n = loss count

    # Initialize output arrays
    occ_recov_before_agg_deduct = np.empty(n, dtype="int64")
    agg_deduct_before_occ = np.empty(n, dtype="int64")
    occ_recov_after_agg_deduct = np.empty(n, dtype="int64")
    agg_deduct_after_occ = np.empty(n, dtype="int64")
    agg_limit_before_occ = np.empty(n, dtype="int64")
    ceded = np.empty(n, dtype="int64")
    cumulative_ceded = np.empty(n, dtype="int64")
    agg_limit_after_occ = np.empty(n, dtype="int64")
    net = np.empty(n, dtype="int64")

    for i in range(n):
        occ_recov_before_agg_deduct[i] = min(occ_limit, max(0, gross[i] - occ_deduct))
        agg_deduct_before_occ[i] = (
            agg_deduct
            if (i == 0 or year[i] != year[i - 1])
            else agg_deduct_after_occ[i - 1]
        )
        occ_recov_after_agg_deduct[i] = max(
            0, occ_recov_before_agg_deduct[i] - agg_deduct_before_occ[i]
        )
        agg_deduct_after_occ[i] = max(
            0, agg_deduct_before_occ[i] - occ_recov_before_agg_deduct[i]
        )
        agg_limit_before_occ[i] = (
            agg_limit
            if (i == 0 or year[i] != year[i - 1])
            else agg_limit_after_occ[i - 1]
        )
        ceded[i] = min(occ_recov_after_agg_deduct[i], agg_limit_before_occ[i])
        cumulative_ceded[i] = (
            ceded[i]
            if (i == 0 or year[i] != year[i - 1])
            else cumulative_ceded[i - 1] + ceded[i]
        )
        agg_limit_after_occ[i] = max(0, agg_limit_before_occ[i] - ceded[i])
        net[i] = gross[i] - ceded[i]

    return (
        occ_recov_before_agg_deduct,
        agg_deduct_before_occ,
        occ_recov_after_agg_deduct,
        agg_deduct_after_occ,
        agg_limit_before_occ,
        ceded,
        cumulative_ceded,
        agg_limit_after_occ,
        net,
    )


@njit
def get_reinst_limits(reinst_number, agg_limit, occ_limit):
    n = len(reinst_number)  # n = reinstatement count

    reinst_limit_before_agg_limit = np.empty(n, dtype="int64")
    reinst_deduct = np.empty(n, dtype="int64")
    reinst_limit_after_agg_limit = np.empty(n, dtype="int64")

    for i in range(n):
        reinst_limit_before_agg_limit[i] = reinst_number[i] * occ_limit
        reinst_deduct[i] = (
            0
            if (i == 0)
            else reinst_deduct[i - 1] + reinst_limit_before_agg_limit[i - 1]
        )
        reinst_limit_after_agg_limit[i] = min(
            reinst_limit_before_agg_limit[i],
            max(0, (agg_limit - occ_limit) - reinst_deduct[i]),
        )

    return reinst_limit_before_agg_limit, reinst_deduct, reinst_limit_after_agg_limit


@njit
def get_reinst_factors(
    ceded_by_year, occ_limit, reinst_rate, reinst_deduct, reinst_limit
):
    years_count = len(ceded_by_year)
    reinst_count = len(reinst_rate)

    # Initialize the temporary variable
    additional_premium = np.empty((years_count, reinst_count), dtype="float64")

    # Initialize the output variable reinst_factor
    reinst_factor = np.empty(years_count, dtype="float64")

    for i in range(years_count):
        for j in range(reinst_count):
            additional_premium[i, j] = (
                min(reinst_limit[j], max(0, ceded_by_year[i] - reinst_deduct[j]))
                * reinst_rate[j]
                / occ_limit
            )
        reinst_factor[i] = 1 + additional_premium[i].sum()

    return reinst_factor


@njit
def get_occ_reinstatements(
    year,
    cumulative_ceded,
    occ_limit,
    reinst_rate,
    reinst_deduct,
    reinst_limit,
    paid_premium,
):
    loss_count = len(cumulative_ceded)
    reinst_count = len(reinst_rate)

    # Initialize the temporary variables. dtype="float64" is suitable here
    reinst_limit_before_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reinst_deduct_before_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reinstated_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reinst_limit_after_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reins_deduct_after_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reinst_premium_occ = np.empty((loss_count, reinst_count), dtype="float64")

    # Initialize the output variables
    reinstated = np.empty(loss_count, dtype="int64")
    reinst_premium = np.empty(loss_count, dtype="int64")

    for i in range(loss_count):
        for j in range(reinst_count):
            reinst_limit_before_occ[i, j] = (
                reinst_limit[j]
                if (i == 0 or year[i] != year[i - 1])
                else reinst_limit_after_occ[i - 1, j]
            )
            reinst_deduct_before_occ[i, j] = (
                reinst_deduct[j]
                if (i == 0 or year[i] != year[i - 1])
                else reins_deduct_after_occ[i - 1, j]
            )
            reinstated_occ[i, j] = min(
                reinst_limit_before_occ[i, j],
                max(0, cumulative_ceded[i] - reinst_deduct_before_occ[i, j]),
            )
            reinst_limit_after_occ[i, j] = max(
                0, reinst_limit_before_occ[i, j] - reinstated_occ[i, j]
            )
            reins_deduct_after_occ[i, j] = (
                reinst_deduct_before_occ[i, j] + reinstated_occ[i, j]
                if (i == 0 or year[i] != year[i - 1])
                else reins_deduct_after_occ[i - 1, j] + reinstated_occ[i, j]
            )
            reinst_premium_occ[i, j] = (
                reinstated_occ[i, j] / occ_limit * reinst_rate[j] * paid_premium
            )
        reinstated[i] = reinstated_occ[i].sum()
        reinst_premium[i] = reinst_premium_occ[i].sum()

    return reinstated, reinst_premium
