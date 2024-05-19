import numpy as np
import pandas as pd
import structlog
from numba import njit
from numpy import ndarray
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import Analysis, HistoLoss, Layer, LayerReinstatement, Premium

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
        log.warning("Analysis not found", analysis_id=analysis_id)
        return pd.DataFrame()

    log.info("Processing analysis", analysis_id=analysis_id)
    burningcosts = [
        get_df_burningcost_for_layer(layer, start_year, end_year, session)
        for layer in analysis.layers
    ]
    return pd.concat(burningcosts, ignore_index=True)


def get_df_burningcost_for_layer(
    layer: Layer, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Calculate the burning cost for a given layer over a range of years.

    :param layer: The layer for which to calculate the burning cost.
    :param start_year: The start year for the calculation.
    :param end_year: The end year for the calculation.
    :param session: Database session for data retrieval.
    :return: A DataFrame containing the burning cost for 'as_is' and 'as_if' bases.
    """
    log.info("Calculating burning cost for layer", layer=layer)
    burningcosts = [
        get_df_burning_cost_for_basis(layer, basis, start_year, end_year, session)
        for basis in ["as_is", "as_if"]
    ]
    return pd.concat(burningcosts, ignore_index=True)


def get_df_burning_cost_for_basis(
    layer: Layer, basis: str, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Calculate the burning cost for a given basis over a range of years.

    :param layer: The layer for which to calculate the burning cost.
    :param basis: The basis type ('as_is' or 'as_if').
    :param start_year: The start year for the calculation.
    :param end_year: The end year for the calculation.
    :param session: Database session for data retrieval.
    :return: A DataFrame containing the burning cost for the specified basis.
    """
    log.info("Calculating burning cost for basis", basis=basis)
    df_burningcost = pd.DataFrame(
        {
            "layer_id": layer.id,
            "basis": basis,
            "year": np.arange(start_year, end_year + 1),
        }
    )

    df_premium_by_year = get_df_premium_by_year(
        layer, basis, start_year, end_year, session
    )
    df_burningcost = pd.merge(
        df_burningcost, df_premium_by_year, how="outer", on="year"
    ).fillna(0)

    df_loss_by_year = get_df_loss_by_year(layer, basis, start_year, end_year, session)
    df_burningcost = pd.merge(
        df_burningcost, df_loss_by_year, how="outer", on="year"
    ).fillna(0)

    return df_burningcost


def get_df_premium_by_year(
    layer: Layer,
    basis: str,
    start_year: int,
    end_year: int,
    session: Session,
) -> pd.DataFrame:
    """
    Retrieve a DataFrame with aggregated premium amounts for a specific layer, basis, and year range.

    :param layer: The layer for which the premiums are retrieved.
    :param basis: Specifies the premium basis: either "as_is" or "as_if".
    :param start_year: The start year for the premium selection.
    :param end_year: The end year for the premium selection.
    :param session: The SQLAlchemy session for database interaction.
    :return: A DataFrame containing two columns: 'year' and 'premium' for the specified basis aggregated by year.
    :raises ValueError: If the `basis` is not one of the allowed values.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    premiumfile_ids = [premiumfile.id for premiumfile in layer.premiumfiles]

    if not premiumfile_ids:
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
    layer: Layer, basis: str, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Retrieve and process loss data by year for a given layer.

    :param layer: The layer for which the loss data is being processed.
    :param basis: Specifies the premium basis: either "as_is" or "as_if".
    :param start_year: The starting year for the loss data.
    :param end_year: The ending year for the loss data.
    :param session: The SQLAlchemy session for database interaction.
    :return: A DataFrame containing the processed loss data by year with columns ['year', 'ceded_before_agg_limits', 'ceded', 'reinstated']. Returns an empty DataFrame with the specified columns if no loss data is found.
    :raises ValueError: If the `basis` is not one of the allowed values.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    # Retrieve individual loss occurences in df_loss
    df_loss = get_df_loss(layer, basis, start_year, end_year, session)

    if df_loss.empty:
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

    df_reinst = get_df_reinst(layer.id, session)

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
        df_loss["reinstated"] = get_occ_reinstatements(
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
    layer: Layer, basis: str, start_year: int, end_year: int, session: Session
) -> pd.DataFrame:
    """
    Retrieve a DataFrame with the losses for a specific layer, basis and year range.

    :param layer: The layer for which the losses are retrieved.
    :param basis: Specifies the loss basis: either "as_is" or "as_if".
    :param start_year: The start year for the loss selection.
    :param end_year: The end year for the loss selection.
    :param session: The SQLAlchemy session for database connection.
    :return: A DataFrame containing two columns: 'year' and 'gross'
    :raises ValueError: If the `basis` is not one of the allowed values.
    """
    if basis not in ["as_is", "as_if"]:
        raise ValueError('basis must be "as_is" or "as_if"')

    histolossfile_ids = [histolossfile.id for histolossfile in layer.histolossfiles]

    if not histolossfile_ids:
        return pd.DataFrame(columns=["year", "gross"])

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


@njit
def get_occ_recoveries(
    year: ndarray,
    gross: ndarray,
    occ_limit: int,
    occ_deduct: int,
    agg_limit: int,
    agg_deduct: int,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Calculate recovery amounts from loss occurrences under specified limits and deductibles.

    This function computes recoveries and net amounts for losses based on occurrence and
    aggregate limits and deductibles. It processes an array of gross loss amounts for given years and
    determines the recoverable and net amounts after applying these deductibles and limits.

    The function uses Numba's njit decorator to improve performance.

    :param year: Array of integers representing the years for each loss.
    :param gross: Array of floats representing the gross loss amounts for each occurrence.
    :param occ_limit: The maximum amount recoverable for any single occurrence.
    :param occ_deduct: The deductible amount applied to each individual occurrence.
    :param agg_limit: The aggregate limit across all occurrences within the same year.
    :param agg_deduct: The deductible amount that applies to all occurrences combined within the same year.
    :return: A tuple containing three ndarrays:
              1. Occurrence recoveries before applying the aggregate deductible.
              2. Ceded amounts after applying both occurrence and aggregate calculations.
              3. Cumulative ceded amounts for successive losses within the same year.
              4. The cede loss count
    """
    n = len(gross)  # n = loss count

    # Initialize arrays for storing calculations
    occ_recov_before_agg_deduct = np.empty(n, dtype=np.int64)
    agg_deduct_before_occ = np.empty(n, dtype=np.int64)
    occ_recov_after_agg_deduct = np.empty(n, dtype=np.int64)
    agg_deduct_after_occ = np.empty(n, dtype=np.int64)
    agg_limit_before_occ = np.empty(n, dtype=np.int64)
    ceded = np.empty(n, dtype=np.int64)
    cumulative_ceded = np.empty(n, dtype=np.int64)
    ceded_loss_count = np.empty(n, dtype=np.int64)
    agg_limit_after_occ = np.empty(n, dtype=np.int64)
    net = np.empty(n, dtype=np.int64)

    for i in range(n):
        occ_recov_before_agg_deduct[i] = min(
            occ_limit, max(0, int(gross[i] - occ_deduct))
        )
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
        ceded_loss_count[i] = 1 if ceded[i] > 0 else 0
        cumulative_ceded[i] = (
            ceded[i]
            if (i == 0 or year[i] != year[i - 1])
            else cumulative_ceded[i - 1] + ceded[i]
        )
        agg_limit_after_occ[i] = max(0, int(agg_limit_before_occ[i] - ceded[i]))
        net[i] = gross[i] - ceded[i]

    return occ_recov_before_agg_deduct, ceded, ceded_loss_count, cumulative_ceded


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


@njit
def get_reinst_limits(
    reinst_number: ndarray, agg_limit: int, occ_limit: int
) -> tuple[ndarray, ndarray]:
    """
    Calculate the deductible and the remaining reinstatement limit after the aggregate limit.

    This function calculates two main components for each reinstatement: the cumulative deductible
    up to that reinstatement and the remaining limit after considering the aggregate limit. The
    calculation uses the reinstatement number, which multiplies the occurrence limit to form a
    reinstatement limit before the aggregate limit is applied. It then calculates the cumulative
    deductible and the remaining reinstatement limit, which are affected by the aggregate and
    occurrence limits.

    This function is optimized with Numba's njit decorator for performance.

    :param reinst_number: An array of reinstatement multipliers, indicating how many times the occurrence limit
                          is applied to calculate the preliminary reinstatement limit.
    :param agg_limit: The total aggregate limit that affects all reinstatements collectively.
    :param occ_limit: The individual occurrence limit applied per reinstatement.
    :return: A tuple of two ndarrays:
             1. The cumulative deductible applied up to each reinstatement.
             2. The remaining limit for each reinstatement after considering the aggregate limit.
    """
    n = len(reinst_number)  # n = reinstatement count

    # Initialize arrays for storing calculations
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
            max(0, int((agg_limit - occ_limit) - reinst_deduct[i])),
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
    """
    Calculate additional premiums based on ceded amounts, occurrence limits, reinstatement rates,
    deductibles, and limits for each year and each reinstatement.

    This function computes the additional premium for each year considering the ceded amounts and
    applying the rates, deductibles, and limits associated with each reinstatement. Premiums are
    adjusted based on the proportion of the ceded amount that exceeds the deductible up to the
    maximum limit, then multiplied by the reinstatement rate and normalized by the occurrence limit.

    The function leverages Numba's njit decorator to optimize performance.

    :param ceded_by_year: An array of ceded amounts for each year.
    :param occ_limit: The occurrence limit that normalizes the calculation of additional premiums.
    :param reinst_rate: An array of rates corresponding to each reinstatement.
    :param reinst_deduct: An array of deductible amounts corresponding to each reinstatement.
    :param reinst_limit: An array of limits corresponding to each reinstatement, setting the maximum claimable
                         amount for additional premiums.
    :return: An array containing the total additional premium calculated for each year.
    """
    years_count = len(ceded_by_year)
    reinst_count = len(reinst_rate)

    # Initialize arrays for storing calculations
    additional_premium_reinst = np.empty((years_count, reinst_count), dtype=np.int64)
    additional_premium = np.empty(years_count, dtype=np.int64)

    for i in range(years_count):
        for j in range(reinst_count):
            additional_premium_reinst[i, j] = (
                min(
                    int(reinst_limit[j]),
                    max(0, int(ceded_by_year[i] - reinst_deduct[j])),
                )
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
    """
    Calculate the reinstated amounts from cumulative ceded losses under specified reinstatement conditions.

    This function computes the reinstated amounts for losses based on the occurrence limit,
    reinstatement rates, deductibles, and limits. It processes arrays of cumulative ceded losses
    and determines the reinstated amounts and premiums after applying these reinstatement conditions.

    The function uses Numba's njit decorator to improve performance.

    :param year: Array of integers representing the years for each loss.
    :param cumulative_ceded: Array of floats representing the cumulative ceded loss amounts for each occurrence.
    :param occ_limit: The maximum amount recoverable for any single occurrence.
    :param reinst_rate: Array of floats representing the reinstatement rates for each reinstatement.
    :param reinst_deduct: Array of floats representing the deductibles for each reinstatement.
    :param reinst_limit: Array of floats representing the limits for each reinstatement.
    :param paid_premium: The paid premium amount used to calculate reinstatement premiums.
    :return: An array of floats representing the reinstated amounts for each occurrence.
    """
    loss_count = len(cumulative_ceded)
    reinst_count = len(reinst_rate)

    # Initialize arrays for storing calculations
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
                max(0, int(cumulative_ceded[i] - reinst_deduct_before_occ[i, j])),
            )
            reinst_limit_after_occ[i, j] = max(
                0, int(reinst_limit_before_occ[i, j] - reinstated_occ[i, j])
            )
            reins_deduct_after_occ[i, j] = (
                int(reinst_deduct_before_occ[i, j] + reinstated_occ[i, j])
                if (i == 0 or year[i] != year[i - 1])
                else reins_deduct_after_occ[i - 1, j] + reinstated_occ[i, j]
            )
            reinst_premium_occ[i, j] = (
                reinstated_occ[i, j] / occ_limit * reinst_rate[j] * paid_premium
            )
        reinstated[i] = reinstated_occ[i].sum()
        reinst_premium[i] = reinst_premium_occ[i].sum()

    return reinstated
