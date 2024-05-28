import numpy as np
import pandas as pd
import structlog

# from numba import njit
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import Analysis, Layer, LayerReinstatement, ModelYearLoss

pd.set_option("display.max_columns", None)
log = structlog.get_logger()


def get_df_yearloss(
    session: Session, analysis_id: int, simulated_years: int
) -> pd.DataFrame:
    """
    Retrieve the year loss data for a specified analysis and concatenate the results.

    This function fetches the analysis from the database using the provided session and analysis ID.
    It processes the analysis by retrieving the year loss data for each layer within the analysis
    over the specified number of simulated years. The results are concatenated into a single DataFrame.

    :param session: SQLAlchemy session for database access.
    :param analysis_id: The ID of the analysis to retrieve.
    :param simulated_years: The number of simulated years for the analysis.
    :return: A DataFrame containing the concatenated year loss data for the analysis.
    """
    analysis = session.get(Analysis, analysis_id)

    if analysis is None:
        log.warning("Analysis not found")
        return pd.DataFrame()

    log.info("Processing analysis", analysis_id=analysis_id)
    layeryearlosses = [
        get_df_layeryearloss(session, layer.id, simulated_years)
        for layer in analysis.layers
    ]
    return pd.concat(layeryearlosses, ignore_index=True)


def get_df_layeryearloss(
    session: Session, layer_id: int, simulated_years: int
) -> pd.DataFrame:
    """
    Calculate and return the year loss data for a specified layer over a given number of simulated years.

    This function retrieves the layer from the database using the provided session and layer ID.
    It calculates the year losses for the layer, processes recoveries, and handles reinstatements.
    The results are returned in a DataFrame.

    :param session: SQLAlchemy session for database access.
    :param layer_id: The ID of the layer to retrieve.
    :param simulated_years: The number of simulated years for the analysis.
    :return: A DataFrame containing the year loss data for the specified layer.
    """
    log.info("Calculating year losses for layer", layer_id=layer_id)

    layer = session.get(Layer, layer_id)
    if layer is None:
        log.warning(f"Layer with ID {layer_id} not found.")
        return pd.DataFrame()

    df = get_df_modelyearloss(session, [modelfile.id for modelfile in layer.modelfiles])
    df["layer_id"] = layer_id

    # Process recoveries
    (
        df["ceded_before_agg_limits"],
        df["ceded"],
        df["ceded_loss_count"],
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
    log.info("expected_annual_loss", expected_annual_loss=expected_annual_loss)

    df_reinst = get_df_reinst(session, layer_id)
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
        log.info("paid_premium", paid_premium=paid_premium)

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


def get_df_modelyearloss(session: Session, modelfile_ids: list[int]) -> pd.DataFrame:
    """
    Retrieve and return model year loss data for the specified model file IDs.

    This function queries the database for model year loss data corresponding to the given list of model file IDs.
    The results are ordered by year and day and returned in a DataFrame.

    :param session: The SQLAlchemy Session for database operations.
    :param modelfile_ids: List of model file IDs to retrieve the year loss data for.
    :return: A DataFrame containing the model year loss data.
    """
    query = (
        select(
            ModelYearLoss.year,
            ModelYearLoss.day,
            ModelYearLoss.loss.label("gross"),
            ModelYearLoss.loss_type,
        )
        .where(ModelYearLoss.modelfile_id.in_(modelfile_ids))
        .order_by(ModelYearLoss.year, ModelYearLoss.day)
    )
    return pd.read_sql_query(query, session.connection())


def get_df_reinst(session: Session, layer_id: int) -> pd.DataFrame:
    """
    Retrieve and return the reinstatement data for a specified layer.

    This function queries the database for reinstatement data corresponding to the given layer ID.
    The results are ordered by the reinstatement order and returned in a DataFrame.

    :param session: The SQLAlchemy Session for database operations.
    :param layer_id: The ID of the layer to retrieve the reinstatement data for.
    :return: A DataFrame containing the reinstatement data for the specified layer.
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
    return pd.read_sql_query(query, session.connection())


def get_occ_recoveries(
    year: np.ndarray,
    gross: np.ndarray,
    occ_limit: int,
    occ_deduct: int,
    agg_limit: int,
    agg_deduct: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate recovery amounts from loss occurrences under specified limits and deductibles.

    This function computes recoveries and net amounts for losses based on occurrence and
    aggregate limits and deductibles. It processes an array of gross loss amounts for given years and
    determines the recoverable and net amounts after applying these deductibles and limits.

    :param year: Array of integers representing the years for each loss.
    :param gross: Array of floats representing the gross loss amounts for each occurrence.
    :param occ_limit: The maximum amount recoverable for any single occurrence.
    :param occ_deduct: The deductible amount applied to each individual occurrence.
    :param agg_limit: The aggregate limit across all occurrences within the same year.
    :param agg_deduct: The deductible amount that applies to all occurrences combined within the same year.
    :return: A tuple containing four ndarrays:
              1. Occurrence recoveries before applying the aggregate deductible.
              2. Ceded amounts after applying both occurrence and aggregate calculations.
              3. The count of the ceded losses.
              4. Cumulative ceded amounts for successive losses within the same year.
              5. Net amounts after cession.
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

    return occ_recov_before_agg_deduct, ceded, ceded_loss_count, cumulative_ceded, net


def get_reinst_limits(
    reinst_number: np.ndarray, agg_limit: int, occ_limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the deductible and the remaining reinstatement limit after the aggregate limit.

    This function calculates two main components for each reinstatement: the cumulative deductible
    up to that reinstatement and the remaining limit after considering the aggregate limit. The
    calculation uses the reinstatement number, which multiplies the occurrence limit to form a
    reinstatement limit before the aggregate limit is applied. It then calculates the cumulative
    deductible and the remaining reinstatement limit, which are affected by the aggregate and
    occurrence limits.

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


def get_additional_premiums(
    ceded_by_year: np.ndarray,
    occ_limit: int,
    reinst_rate: np.ndarray,
    reinst_deduct: np.ndarray,
    reinst_limit: np.ndarray,
) -> np.ndarray:
    """
    Calculate additional premiums based on ceded amounts, occurrence limits, reinstatement rates,
    deductibles, and limits for each year and each reinstatement.

    This function computes the additional premium for each year considering the ceded amounts and
    applying the rates, deductibles, and limits associated with each reinstatement. Premiums are
    adjusted based on the proportion of the ceded amount that exceeds the deductible up to the
    maximum limit, then multiplied by the reinstatement rate and normalized by the occurrence limit.

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


def get_occ_reinstatements(
    year: np.ndarray,
    cumulative_ceded: np.ndarray,
    occ_limit: int,
    reinst_rate: np.ndarray,
    reinst_deduct: np.ndarray,
    reinst_limit: np.ndarray,
    paid_premium: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the reinstated amounts and premiums from cumulative ceded losses under specified reinstatement conditions.

    This function computes the reinstated amounts for losses based on the occurrence limit,
    reinstatement rates, deductibles, and limits. It processes arrays of cumulative ceded losses
    and determines the reinstated amounts and premiums after applying these reinstatement conditions.

    :param year: Array of integers representing the years for each loss.
    :param cumulative_ceded: Array of floats representing the cumulative ceded loss amounts for each occurrence.
    :param occ_limit: The maximum amount recoverable for any single occurrence.
    :param reinst_rate: Array of floats representing the reinstatement rates for each reinstatement.
    :param reinst_deduct: Array of floats representing the deductibles for each reinstatement.
    :param reinst_limit: Array of floats representing the limits for each reinstatement.
    :param paid_premium: The paid premium amount used to calculate reinstatement premiums.
    :return:
        - An array of floats representing the reinstated amounts for each occurrence.
        - An array of floats representing the reinstated premiums for each occurrence.
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

    return reinstated, reinst_premium
