import numpy as np
import pandas as pd
from numba import njit
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import Layer, LayerReinstatement, ModelYearLoss, engine

pd.set_option("display.max_columns", None)


def get_df_layeryearloss(layer_id, modelfiles_ids, simulated_years):
    layer = get_layer(layer_id)
    df = get_df_modelyearloss(modelfiles_ids)
    df["layer_id"] = layer_id

    # Process recoveries
    (
        df["ceded"],
        df["cumulative_ceded"],
        df["net"],
    ) = get_occ_recoveries(
        df["year"].to_numpy(),
        df["gross"].to_numpy(),
        layer.occ_limit,
        layer.occ_deduct,
        layer.agg_limit,
        layer.agg_deduct,
    )

    # Initialize reinstated and reinst_premium to 0
    (df["reinstated"], df["reinst_premium"]) = (0, 0)

    # Process reinstatements
    df_by_year = df[["year", "ceded"]].groupby(by="year").sum()
    expected_annual_loss = df_by_year["ceded"].sum() / simulated_years
    print(f"{expected_annual_loss=:,.0f}")

    df_reinst = get_df_reinst(layer_id)
    if not df_reinst.empty:
        (df_reinst["deduct"], df_reinst["limit"]) = get_reinst_limits(
            df_reinst["number"].to_numpy(), layer.agg_limit, layer.occ_limit
        )

        df_by_year["additional_premium"] = get_additional_premiums(
            df_by_year["ceded"].to_numpy(),
            layer.occ_limit,
            df_reinst["rate"].to_numpy(),
            df_reinst["deduct"].to_numpy(),
            df_reinst["limit"].to_numpy(),
        )

        paid_premium = expected_annual_loss / (
            1 + df_by_year["additional_premium"].sum() / simulated_years
        )
        print(f"{paid_premium=:,.0f}")

        (df["reinstated"], df["reinst_premium"]) = get_occ_reinstatements(
            df["year"].to_numpy(),
            df["cumulative_ceded"].to_numpy(),
            layer.occ_limit,
            df_reinst["rate"].to_numpy(),
            df_reinst["deduct"].to_numpy(),
            df_reinst["limit"].to_numpy(),
            paid_premium,
        )

    # Finally
    df = df[
        [
            "year",
            "day",
            "gross",
            "ceded",
            "net",
            "reinstated",
            "reinst_premium",
            "loss_type",
            "layer_id",
        ]
    ]

    return df


def get_linked_modelfiles(layer_id, df_layer_modelfile):
    return df_layer_modelfile[df_layer_modelfile["layer_id"] == layer_id][
        "modelfile_id"
    ]


def get_layer(layer_id):
    return Session(engine).get(Layer, layer_id)


def get_df_modelyearloss(modelfile_ids):
    query = (
        select(ModelYearLoss)
        .filter(ModelYearLoss.modelfile_id.in_(modelfile_ids))
        .order_by(ModelYearLoss.year, ModelYearLoss.day)
    )
    df = pd.read_sql_query(query, engine)
    df = df.drop(columns=["id"])
    df = df.rename(columns={"loss": "gross"})
    return df


def get_df_reinst(layer_id):
    query = (
        select(LayerReinstatement)
        .filter_by(layer_id=layer_id)
        .order_by(LayerReinstatement.order)
    )
    df = pd.read_sql_query(query, engine)
    df = df.drop(columns=["id", "layer_id"])
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

    # Initialize arrays
    occ_recov_before_agg_deduct = np.empty(n, dtype=np.int64)
    agg_deduct_before_occ = np.empty(n, dtype=np.int64)
    occ_recov_after_agg_deduct = np.empty(n, dtype=np.int64)
    agg_deduct_after_occ = np.empty(n, dtype=np.int64)
    agg_limit_before_occ = np.empty(n, dtype=np.int64)
    ceded = np.empty(n, dtype=np.int64)
    cumulative_ceded = np.empty(n, dtype=np.int64)
    agg_limit_after_occ = np.empty(n, dtype=np.int64)
    net = np.empty(n, dtype=np.int64)

    for i in range(n):
        occ_recov_before_agg_deduct[i] = min(occ_limit, max(0, gross[i] - occ_deduct))
        agg_deduct_before_occ[i] = (
            agg_deduct
            if (i == 0 or year[i] != year[i - 1])
            else agg_deduct_after_occ[i - 1]
        )
        occ_recov_after_agg_deduct[i] = max(
            0, int(occ_recov_before_agg_deduct[i] - agg_deduct_before_occ[i])
        )
        agg_deduct_after_occ[i] = max(
            0, int(agg_deduct_before_occ[i] - occ_recov_before_agg_deduct[i])
        )
        agg_limit_before_occ[i] = (
            agg_limit
            if (i == 0 or year[i] != year[i - 1])
            else agg_limit_after_occ[i - 1]
        )
        ceded[i] = min(int(occ_recov_after_agg_deduct[i]), int(agg_limit_before_occ[i]))
        cumulative_ceded[i] = (
            ceded[i]
            if (i == 0 or year[i] != year[i - 1])
            else cumulative_ceded[i - 1] + ceded[i]
        )
        agg_limit_after_occ[i] = max(0, int(agg_limit_before_occ[i] - ceded[i]))
        net[i] = gross[i] - ceded[i]

    return ceded, cumulative_ceded, net


@njit
def get_reinst_limits(reinst_number, agg_limit, occ_limit):
    n = len(reinst_number)  # n = reinstatement count

    reinst_limit_before_agg_limit = np.empty(n, dtype=np.int64)
    reinst_deduct = np.empty(n, dtype=np.int64)
    reinst_limit_after_agg_limit = np.empty(n, dtype=np.int64)

    for i in range(n):
        reinst_limit_before_agg_limit[i] = reinst_number[i] * occ_limit
        reinst_deduct[i] = (
            0
            if (i == 0)
            else reinst_deduct[i - 1] + reinst_limit_before_agg_limit[i - 1]
        )
        reinst_limit_after_agg_limit[i] = min(
            int(reinst_limit_before_agg_limit[i]),
            max(0, (agg_limit - occ_limit) - reinst_deduct[i]),
        )

    return reinst_deduct, reinst_limit_after_agg_limit


@njit
def get_additional_premiums(
    ceded_by_year, occ_limit, reinst_rate, reinst_deduct, reinst_limit
):
    years_count = len(ceded_by_year)
    reinst_count = len(reinst_rate)

    additional_premium_reinst = np.empty((years_count, reinst_count), dtype=np.int64)
    additional_premium = np.empty(years_count, dtype=np.int64)

    for i in range(years_count):
        for j in range(reinst_count):
            additional_premium_reinst[i, j] = (
                min(reinst_limit[j], max(0, ceded_by_year[i] - reinst_deduct[j]))
                * reinst_rate[j]
                / occ_limit
            )
        additional_premium[i] = additional_premium_reinst[i].sum()

    return additional_premium


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

    reinst_limit_before_occ = np.empty((loss_count, reinst_count), dtype=np.int64)
    reinst_deduct_before_occ = np.empty((loss_count, reinst_count), dtype=np.int64)
    reinstated_occ = np.empty((loss_count, reinst_count), dtype=np.int64)
    reinst_limit_after_occ = np.empty((loss_count, reinst_count), dtype=np.int64)
    reins_deduct_after_occ = np.empty((loss_count, reinst_count), dtype=np.int64)
    reinst_premium_occ = np.empty((loss_count, reinst_count), dtype=np.int64)

    reinstated = np.empty(loss_count, dtype=np.int64)
    reinst_premium = np.empty(loss_count, dtype=np.int64)

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
                int(reinst_limit_before_occ[i, j]),
                max(0, cumulative_ceded[i] - reinst_deduct_before_occ[i, j]),
            )
            reinst_limit_after_occ[i, j] = max(
                0, int(reinst_limit_before_occ[i, j] - reinstated_occ[i, j])
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
