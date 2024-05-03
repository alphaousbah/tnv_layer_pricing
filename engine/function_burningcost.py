from typing import Optional, Literal, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray
from numba import njit
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from db import Analysis, Layer, LayerReinstatement, Premium, HistoLoss, engine

pd.set_option("display.max_columns", None)


def get_df_burningcost(
    analysis_id: int, start_year: int, end_year: int
) -> Optional[pd.DataFrame]:
    # TODO: Think of the better way to process code
    with Session(engine) as session:
        analysis = session.get(Analysis, analysis_id)

        for layer in analysis.layers:
            df_layer_burningcost = get_df_layer_burningcost(
                layer, start_year, end_year, session
            )

    return None


def get_df_layer_burningcost(
    layer: Layer, start_year: int, end_year: int, session: Session
) -> Optional[pd.DataFrame]:
    print(f"{layer.id=}\n")

    # TODO: Concatenate dataframes !!!!!!!!!!!!!!!!!!!!!!
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    for basis in ["as_is", "as_if"]:
        # Initialize df_burningcost
        df = pd.DataFrame({"basis": basis, "year": np.arange(start_year, end_year + 1)})

        # Get premiums
        df_premium = get_df_premium(layer, basis, start_year, end_year, session)
        df = pd.merge(
            df, df_premium, how="outer", on="year"
        ).fillna(0)

        # Get losses and recoveries
        df_loss = get_df_loss(layer, basis, start_year, end_year, session)

        # Initialize ceded_before_agg_limits, ceded and reinstated to 0
        df_loss["ceded_before_agg_limits"] = 0
        df_loss["ceded"] = 0
        df_loss["reinstated"] = 0

        if not df_loss.empty:
            # Process recoveries
            (
                df_loss["ceded_before_agg_limits"],
                df_loss["ceded"],
                df_loss["cumulative_ceded"],
            ) = get_occ_recoveries(
                df_loss["year"].to_numpy(),
                df_loss["gross"].to_numpy(),
                layer.occ_limit,
                layer.occ_deduct,
                layer.agg_limit,
                layer.agg_deduct,
            )

            # Process reinstatements
            df_by_year = (
                df_loss[["year", "ceded"]].groupby(by="year").sum()
            )
            expected_annual_loss = df_by_year["ceded"].mean()
            print(f"{expected_annual_loss=:,.0f}")

            df_reinst = get_df_reinst(layer.id, session)
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
                    1 + df_by_year["additional_premium"].mean()
                )
                print(f"{paid_premium=:,.0f}")

                df_loss["reinstated"] = get_occ_reinstatements(
                    df_loss["year"].to_numpy(),
                    df_loss["cumulative_ceded"].to_numpy(),
                    layer.occ_limit,
                    df_reinst["rate"].to_numpy(),
                    df_reinst["deduct"].to_numpy(),
                    df_reinst["limit"].to_numpy(),
                    paid_premium,
                )

    return df


def get_df_premium(
    layer: Layer,
    basis: str,
    start_year: int,
    end_year: int,
    session: Session,
) -> Optional[pd.DataFrame]:
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    # TODO: Add a Group by year in premiums query
    premiumfile_ids = [premiumfile.id for premiumfile in layer.premiumfiles]
    query = (
        select(Premium)
        .where(
            Premium.premiumfile_id.in_(premiumfile_ids),
            Premium.year.in_(np.arange(start_year, end_year + 1)),
        )
        .order_by(Premium.year)
    )
    df = pd.read_sql_query(query, session.get_bind())
    df["premium"] = df[f"{basis}_premium"]
    df = df[["year", "premium"]]
    return df


def get_df_loss(
    layer: Layer, basis: str, start_year: int, end_year: int, session: Session
) -> Optional[pd.DataFrame]:
    histolossfile_ids = [histolossfile.id for histolossfile in layer.histolossfiles]
    query = (
        select(HistoLoss)
        .where(
            HistoLoss.lossfile_id.in_(histolossfile_ids),
            HistoLoss.year.in_(np.arange(start_year, end_year + 1)),
        )
        .order_by(HistoLoss.year)
    )
    df = pd.read_sql_query(query, session.get_bind())
    df["gross"] = df[f"{basis}_loss"]
    df = df[["year", "gross"]]
    return df


@njit
def get_occ_recoveries(
    year: ndarray,
    gross: ndarray,
    occ_limit: int,
    occ_deduct: int,
    agg_limit: int,
    agg_deduct: int,
) -> tuple[ndarray, ndarray, ndarray]:
    n = len(gross)  # n = loss count

    # Initialize arrays
    occ_recov_before_agg_deduct = np.empty(n, dtype="float64")
    agg_deduct_before_occ = np.empty(n, dtype="float64")
    occ_recov_after_agg_deduct = np.empty(n, dtype="float64")
    agg_deduct_after_occ = np.empty(n, dtype="float64")
    agg_limit_before_occ = np.empty(n, dtype="float64")
    ceded = np.empty(n, dtype="float64")
    cumulative_ceded = np.empty(n, dtype="float64")
    agg_limit_after_occ = np.empty(n, dtype="float64")
    net = np.empty(n, dtype="float64")

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

    return occ_recov_before_agg_deduct, ceded, cumulative_ceded


def get_df_reinst(layer_id: int, session: Session) -> Optional[pd.DataFrame]:
    query = (
        select(LayerReinstatement)
        .filter_by(layer_id=layer_id)
        .order_by(LayerReinstatement.order)
    )
    df = pd.read_sql_query(query, session.get_bind())
    df = df.drop(columns=["id", "layer_id"])
    return df


@njit
def get_reinst_limits(
    reinst_number: ndarray, agg_limit: int, occ_limit: int
) -> tuple[ndarray, ndarray]:
    n = len(reinst_number)  # n = reinstatement count

    reinst_limit_before_agg_limit = np.empty(n, dtype="float64")
    reinst_deduct = np.empty(n, dtype="float64")
    reinst_limit_after_agg_limit = np.empty(n, dtype="float64")

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

    return reinst_deduct, reinst_limit_after_agg_limit


@njit
def get_additional_premiums(
    ceded_by_year: ndarray,
    occ_limit: int,
    reinst_rate: ndarray,
    reinst_deduct: ndarray,
    reinst_limit: ndarray,
) -> ndarray:
    years_count = len(ceded_by_year)
    reinst_count = len(reinst_rate)

    additional_premium_reinst = np.empty((years_count, reinst_count), dtype="float64")
    additional_premium = np.empty(years_count, dtype="float64")

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
    year: ndarray,
    cumulative_ceded: ndarray,
    occ_limit: int,
    reinst_rate: ndarray,
    reinst_deduct: ndarray,
    reinst_limit: ndarray,
    paid_premium: float,
) -> ndarray:
    loss_count = len(cumulative_ceded)
    reinst_count = len(reinst_rate)

    reinst_limit_before_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reinst_deduct_before_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reinstated_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reinst_limit_after_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reins_deduct_after_occ = np.empty((loss_count, reinst_count), dtype="float64")
    reinst_premium_occ = np.empty((loss_count, reinst_count), dtype="float64")

    reinstated = np.empty(loss_count, dtype="float64")
    reinst_premium = np.empty(loss_count, dtype="float64")

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

    return reinstated
