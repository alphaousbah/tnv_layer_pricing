import numpy as np
import pandas as pd
import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import (
    Analysis,
    HistoLoss,
    Layer,
    Premium,
    layer_histolossfile,
    layer_premiumfile,
)
from engine.function_layeryearloss import (
    get_additional_premiums,
    get_df_reinst,
    get_occ_recoveries,
    get_occ_reinstatements,
    get_reinst_limits,
)

pd.set_option("display.max_columns", None)
log = structlog.get_logger()


def get_df_burningcost(
    analysis_id: int, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Calculate the burning cost for all layers in a given analysis over a range of years.

    :param analysis_id: The ID of the analysis to retrieve layers from.
    :param start_year: The start year for the calculation.
    :param end_year: The end year for the calculation.
    :param session: The SQLAlchemy session used for database access.
    :return: A DataFrame containing the burning cost for all layers in the analysis.
    """
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        log.warning(f"Analysis with ID {analysis_id} not found.")
        return pd.DataFrame()

    log.info("Processing analysis", analysis_id=analysis_id)
    burningcosts = [
        get_df_burningcost_for_layer(layer.id, start_year, end_year, session)
        for layer in analysis.layers
    ]
    return pd.concat(burningcosts, ignore_index=True)


def get_df_burningcost_for_layer(
    layer_id: int, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Calculate the burning cost for a given layer over a range of years.

    This function computes the burning cost for a specified insurance layer
    over a given time period for both 'as_is' and 'as_if' bases.

    :param layer_id: The ID of the layer for which to calculate the burning cost.
    :param start_year: The starting year for the calculation.
    :param end_year: The ending year for the calculation.
    :param session: The SQLAlchemy session used for database access.
    :return: A DataFrame containing the burning cost for 'as_is' and 'as_if' bases.
    """
    log.info("Calculating burning cost for layer", layer_id=layer_id)
    burningcosts = [
        get_df_burning_cost_for_basis(layer_id, basis, start_year, end_year, session)
        for basis in ["as_is", "as_if"]
    ]
    return pd.concat(burningcosts, ignore_index=True)


def get_df_burning_cost_for_basis(
    layer_id: int, basis: str, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Calculate the burning cost for a given layer and basis over a range of years.

    This function computes the burning cost for a specified insurance layer
    and basis ('as_is' or 'as_if') over a given time period.

    :param layer_id: The ID of the layer for which to calculate the burning cost.
    :param basis: The basis of the calculation, either 'as_is' or 'as_if'.
    :param start_year: The starting year for the calculation.
    :param end_year: The ending year for the calculation.
    :param session: The SQLAlchemy session used for database access.
    :return: A DataFrame containing the burning cost data.
    """
    log.info("Calculating burning cost for basis", basis=basis)
    df_burningcost = pd.DataFrame(
        {
            "layer_id": layer_id,
            "basis": basis,
            "year": np.arange(start_year, end_year + 1),
            "year_selected": True,
        }
    )

    df_premium_by_year = get_df_premium_by_year(
        layer_id, basis, start_year, end_year, session
    )
    df_burningcost = pd.merge(
        df_burningcost, df_premium_by_year, how="outer", on="year"
    ).fillna(0)

    df_loss_ceded_by_year = get_df_loss_ceded_by_year(
        layer_id, basis, start_year, end_year, session
    )
    df_burningcost = pd.merge(
        df_burningcost, df_loss_ceded_by_year, how="outer", on="year"
    ).fillna(0)

    return df_burningcost


def get_df_premium_by_year(
    layer_id: int,
    basis: str,
    start_year: int,
    end_year: int,
    session: Session,
) -> pd.DataFrame:
    """
    Retrieve the annual premium data for a given layer and basis over a range of years.

    This function fetches the annual premium amounts for a specified insurance layer
    and basis ('as_is' or 'as_if') over a given time period from the database.

    :param layer_id: The ID of the layer for which to retrieve premium data.
    :param basis: The basis of the premium, either 'as_is' or 'as_if'.
    :param start_year: The starting year for the premium data retrieval.
    :param end_year: The ending year for the premium data retrieval.
    :param session: The SQLAlchemy session used for database access.
    :raise ValueError: If the basis is not 'as_is' or 'as_if'.
    :return: A DataFrame containing the annual premium data.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    premium_for_basis = getattr(Premium, f"{basis}_premium")
    premiumfile_ids = select(layer_premiumfile.c.premiumfile_id).where(
        layer_premiumfile.c.layer_id == layer_id
    )

    query = (
        select(
            Premium.year,
            func.sum(premium_for_basis).label("premium"),
        )
        .where(
            Premium.premiumfile_id.in_(premiumfile_ids),
            Premium.year.between(start_year, end_year),
        )
        .group_by(Premium.year)
        .order_by(Premium.year)
    )
    return pd.read_sql_query(query, session.get_bind())


def get_df_loss_ceded_by_year(
    layer_id: int, basis: str, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Retrieve and process loss data by year for a given layer.

    :param layer_id: The ID of the layer for which the loss data is being processed.
    :param basis: Specifies the premium basis: either "as_is" or "as_if".
    :param start_year: The starting year for the loss data.
    :param end_year: The ending year for the loss data.
    :param session: The SQLAlchemy session used for database access.
    :return: A DataFrame containing the processed loss data by year with columns ['year', 'ceded_before_agg_limits', 'ceded', 'ceded_loss_count', 'reinstated']. Returns an empty DataFrame with the specified columns if no loss data is found.
    :raise ValueError: If the `basis` is not one of the allowed values.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    log.info("Processing loss data by year", layer_id=layer_id, basis=basis)

    # Create a DataFrame with default columns for merging with the main DataFrame if no losses are found
    default_df = pd.DataFrame(
        columns=[
            "year",
            "ceded_before_agg_limits",
            "ceded",
            "ceded_loss_count",
            "reinstated",
        ]
    )

    layer = session.get(Layer, layer_id)
    if layer is None:
        log.warning(f"Layer with ID {layer_id} not found.")
        return default_df

    # Retrieve individual losses in df_loss
    df_loss = get_df_loss(layer_id, basis, start_year, end_year, session)
    if df_loss.empty:
        return default_df

    # Process individual recoveries
    (
        df_loss["ceded_before_agg_limits"],
        df_loss["ceded"],
        df_loss["ceded_loss_count"],
        df_loss["cumulative_ceded"],
        df_loss["net"],
    ) = get_occ_recoveries(
        df_loss["year"].to_numpy(),
        df_loss["gross"].to_numpy(),
        layer.occ_limit,
        layer.occ_deduct,
        layer.agg_limit,
        layer.agg_deduct,
    )

    # Process layer reinstatements
    df_stat_by_year = df_loss[["year", "ceded"]].groupby("year").sum()
    expected_annual_loss = df_stat_by_year["ceded"].mean()
    log.info("expected_annual_loss", expected_annual_loss=expected_annual_loss)

    df_reinst = get_df_reinst(layer_id, session)

    if df_reinst.empty:
        df_loss["reinstated"] = 0

    else:
        # Calculate the limits for each reinstatement
        (df_reinst["deduct"], df_reinst["limit"]) = get_reinst_limits(
            df_reinst["number"].to_numpy(), layer.agg_limit, layer.occ_limit
        )

        # Calculate the additional premium for each year based on reinstatements
        df_stat_by_year["additional_premium"] = get_additional_premiums(
            df_stat_by_year["ceded"].to_numpy(),
            layer.occ_limit,
            df_reinst["rate"].to_numpy(),
            df_reinst["deduct"].to_numpy(),
            df_reinst["limit"].to_numpy(),
        )

        # Calculate the paid premium
        paid_premium = expected_annual_loss / (
            1 + df_stat_by_year["additional_premium"].mean()
        )
        log.info("paid_premium", paid_premium=paid_premium)

        # Finally
        # Calculate the reinstated amount for each loss
        # We don't need the reinstated premium here
        (df_loss["reinstated"], df_loss["reinst_premium"]) = get_occ_reinstatements(
            df_loss["year"].to_numpy(),
            df_loss["cumulative_ceded"].to_numpy(),
            layer.occ_limit,
            df_reinst["rate"].to_numpy(),
            df_reinst["deduct"].to_numpy(),
            df_reinst["limit"].to_numpy(),
            paid_premium,
        )

    return (
        df_loss[
            [
                "year",
                "ceded_before_agg_limits",
                "ceded",
                "ceded_loss_count",
                "reinstated",
            ]
        ]
        .groupby("year", as_index=False)
        .sum()
    )


def get_df_loss(
    layer_id: int, basis: str, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Retrieve a DataFrame with the losses for a specific layer, basis and year range.

    :param layer_id: The ID of the layer for which the losses are retrieved.
    :param basis: Specifies the loss basis: either "as_is" or "as_if".
    :param start_year: The starting year for the loss selection.
    :param end_year: The ending year for the loss selection.
    :param session: The SQLAlchemy session used for database access.
    :return: A DataFrame containing two columns: 'year' and 'gross'
    :raise ValueError: If the `basis` is not one of the allowed values.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    loss_for_basis = getattr(HistoLoss, f"{basis}_loss")
    lossfile_ids = select(layer_histolossfile.c.histolossfile_id).where(
        layer_histolossfile.c.layer_id == layer_id
    )

    query = (
        select(HistoLoss.year, loss_for_basis.label("gross"))
        .where(
            HistoLoss.lossfile_id.in_(lossfile_ids),
            HistoLoss.year.between(start_year, end_year),
        )
        .order_by(HistoLoss.year)
    )
    return pd.read_sql_query(query, session.get_bind())
