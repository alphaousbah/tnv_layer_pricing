import numpy as np
import pandas as pd
import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import (
    Analysis,
    HistoLoss,
    Layer,
    LayerReinstatement,
    Premium,
)
from engine.function_layeryearloss import (
    get_additional_premiums,
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
    :param session: Database session for data retrieval.
    :return: A DataFrame containing the burning cost for all layers in the analysis.
    """
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        log.warning(f"Analysis with ID {analysis_id} not found.")
        raise ValueError(f"Analysis with ID {analysis_id} not found.")

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

    :param layer_id: The ID of the layer for which to calculate the burning cost.
    :param start_year: The start year for the calculation.
    :param end_year: The end year for the calculation.
    :param session: Database session for data retrieval.
    :return: A DataFrame containing the burning cost for 'as_is' and 'as_if' bases.
    """
    layer = session.get(Layer, layer_id)
    if layer is None:
        log.warning(f"Layer with ID {layer_id} not found.")
        raise ValueError(f"Layer with ID {layer_id} not found.")

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
    Calculate the burning cost for a given basis over a range of years.

    :param layer_id: The ID of the layer for which to calculate the burning cost.
    :param basis: The basis type ('as_is' or 'as_if').
    :param start_year: The start year for the calculation.
    :param end_year: The end year for the calculation.
    :param session: Database session for data retrieval.
    :return: A DataFrame containing the burning cost for the specified basis.
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

    df_loss_by_year = get_df_loss_by_year(
        layer_id, basis, start_year, end_year, session
    )
    df_burningcost = pd.merge(
        df_burningcost, df_loss_by_year, how="outer", on="year"
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
    Retrieve a DataFrame with aggregated premium amounts for a specific layer, basis, and year range.

    :param layer_id: The ID of the layer for which the premiums are retrieved.
    :param basis: Specifies the premium basis: either "as_is" or "as_if".
    :param start_year: The start year for the premium selection.
    :param end_year: The end year for the premium selection.
    :param session: The SQLAlchemy session for database interaction.
    :return: A DataFrame containing two columns: 'year' and 'premium' for the specified basis aggregated by year.
    :raise ValueError: If the `basis` is not one of the allowed values.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    layer = session.get(Layer, layer_id)
    if layer is None:
        raise ValueError(f"Layer with ID {layer_id} not found.")

    premiumfile_ids = [premiumfile.id for premiumfile in layer.premiumfiles]
    if not premiumfile_ids:
        # Return columns with default values if no premiums are found, for the merge
        return pd.DataFrame(columns=["year", "premium"])

    premium_column = getattr(Premium, f"{basis}_premium")
    query = (
        select(
            Premium.year,
            func.sum(premium_column).label("premium"),
        )
        .where(
            Premium.premiumfile_id.in_(premiumfile_ids),
            Premium.year.between(start_year, end_year),
        )
        .group_by(Premium.year)
        .order_by(Premium.year)
    )
    return pd.read_sql_query(query, session.get_bind())


def get_df_loss_by_year(
    layer_id: int, basis: str, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Retrieve and process loss data by year for a given layer.

    :param layer_id: The ID of the layer for which the loss data is being processed.
    :param basis: Specifies the premium basis: either "as_is" or "as_if".
    :param start_year: The starting year for the loss data.
    :param end_year: The ending year for the loss data.
    :param session: The SQLAlchemy session for database interaction.
    :return: A DataFrame containing the processed loss data by year with columns ['year', 'ceded_before_agg_limits', 'ceded', 'reinstated']. Returns an empty DataFrame with the specified columns if no loss data is found.
    :raise ValueError: If the `basis` is not one of the allowed values.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    layer = session.get(Layer, layer_id)
    if layer is None:
        raise ValueError(f"Layer with ID {layer_id} not found.")

    # Retrieve individual loss occurences in df_loss
    df_loss = get_df_loss(layer_id, basis, start_year, end_year, session)
    if df_loss.empty:
        # Return columns with default values if no losses are found, for the merge
        return pd.DataFrame(
            columns=[
                "year",
                "ceded_before_agg_limits",
                "ceded",
                "ceded_loss_count",
                "reinstated",
            ]
        )

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

    # Process individual reinstatements
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
    :param start_year: The start year for the loss selection.
    :param end_year: The end year for the loss selection.
    :param session: The SQLAlchemy session for database connection.
    :return: A DataFrame containing two columns: 'year' and 'gross'
    :raise ValueError: If the `basis` is not one of the allowed values.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    layer = session.get(Layer, layer_id)
    if layer is None:
        raise ValueError(f"Layer with ID {layer_id} not found.")

    histolossfile_ids = [histolossfile.id for histolossfile in layer.histolossfiles]
    if not histolossfile_ids:
        return pd.DataFrame()

    loss_column = getattr(HistoLoss, f"{basis}_loss")
    query = (
        select(HistoLoss.year, loss_column.label("gross"))
        .where(
            HistoLoss.lossfile_id.in_(histolossfile_ids),
            HistoLoss.year.between(start_year, end_year),
        )
        .order_by(HistoLoss.year)
    )
    return pd.read_sql_query(query, session.get_bind())


def get_df_reinst(layer_id: int, session: Session) -> pd.DataFrame:
    """
    Retrieve a DataFrame with detailed reinstatement information for a specific layer.

    This function queries the database for specific columns of reinstatement data associated with a given layer ID.
    It retrieves the order, number, and rate of reinstatements, sorting them by the order of reinstatement.

    :param layer_id: The unique identifier of the layer for which reinstatement details are to be retrieved.
    :param session: An instance of SQLAlchemy Session to be used for executing the database query.
    :return: A DataFrame containing selected columns ('order', 'number', 'rate') from the LayerReinstatement table
             for the specified layer, sorted by the 'order' column. The DataFrame is empty if no records are found.
    """
    query = (
        select(
            LayerReinstatement.order,
            LayerReinstatement.number,
            LayerReinstatement.rate,
        )
        .where(LayerReinstatement.layer_id == layer_id)
        .order_by(LayerReinstatement.order)
    )
    return pd.read_sql_query(query, session.get_bind())
