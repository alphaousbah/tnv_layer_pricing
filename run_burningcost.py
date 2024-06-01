import sys
from pathlib import Path
from time import perf_counter

import structlog
from sqlalchemy import select
from win32com.client import Dispatch

from database import (
    Analysis,
    HistoLossFile,
    Layer,
    LayerBurningCost,
    PremiumFile,
    Session,
)
from engine.function_burningcost import get_df_burningcost
from utils import (
    df_from_listobject,
    get_single_result,
    read_from_listobject_and_save,
    write_df_in_listobjects,
)

log = structlog.get_logger()

# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = Dispatch("Excel.Application")

try:
    wb_path = sys.argv[1]
    wb = excel.Workbooks.Open(wb_path)
except IndexError:
    wb = excel.Workbooks.Open(f"{Path.cwd()}/run_burningcost.xlsm")

# --------------------------------------
# Step 2: Import the existing DB records
# --------------------------------------

with Session.begin() as session:
    read_from_listobject_and_save(
        session=session,
        worksheet=wb.Worksheets("Database"),
        listobject_names=[
            "Analysis",
            "Layer",
            "LayerReinstatement",
            "PremiumFile",
            "Premium",
            "HistoLossFile",
            "HistoLoss",
        ],
    )

# --------------------------------------
# Step 3: Read the input data
# --------------------------------------

ws_input = wb.Worksheets("Input")

analysis_id = ws_input.Range("analysis_id").value
start_year = ws_input.Range("start_year").value
end_year = ws_input.Range("end_year").value
df_layer_premiumfile = df_from_listobject(ws_input.ListObjects("layer_premiumfile"))
df_layer_histolossfile = df_from_listobject(ws_input.ListObjects("layer_histolossfile"))

# --------------------------------------
# Step 4: Process
# --------------------------------------

start = perf_counter()

with Session.begin() as session:
    analysis = session.get(Analysis, analysis_id)

    if analysis is None:
        log.error(f"Analysis with id {analysis_id} not found")
        raise ValueError(f"Analysis with id {analysis_id} not found")

    # Delete the previous relationships between layers and premiumfiles
    for layer in analysis.layers:
        layer.premiumfiles.clear()

    # Create and save the new relationships between layers and premiumfiles
    for _, row in df_layer_premiumfile.iterrows():
        query_layer = select(Layer).where(Layer.id == row["layer_id"])
        layer = get_single_result(session, query_layer)

        query_premiumfile = select(PremiumFile).where(
            PremiumFile.id == row["premiumfile_id"]
        )
        premiumfile = get_single_result(session, query_premiumfile)

        layer.premiumfiles.append(premiumfile)

    # Delete the previous relationships between layers and histolossfiles
    for layer in analysis.layers:
        layer.histolossfiles.clear()

    # Create and save the new relationships between layers and histolossfiles
    for _, row in df_layer_histolossfile.iterrows():
        query_layer = select(Layer).where(Layer.id == row["layer_id"])
        layer = get_single_result(session, query_layer)

        query_histolossfile = select(HistoLossFile).where(
            HistoLossFile.id == row["histolossfile_id"]
        )
        histolossfile = get_single_result(session, query_histolossfile)

        layer.histolossfiles.append(histolossfile)

    # Delete the previsous burning costs
    for layer in analysis.layers:
        for burningcost in layer.burningcosts:
            session.delete(burningcost)

    # Calculate the burning cost for the analysis and save the results to the database
    df_burningcost = get_df_burningcost(session, analysis_id, start_year, end_year)
    df_burningcost.to_sql(
        name="layerburningcost",
        con=session.connection(),
        if_exists="append",
        index=False,
    )

# --------------------------------------
# Step 5: Write the output data
# --------------------------------------

# Define the output worksheet and table
ws_output = wb.Worksheets("Output")

with Session.begin() as session:
    write_df_in_listobjects(
        session=session,
        DbModels=[LayerBurningCost],
        ws_output=ws_output,
    )

ws_output.Select()
end = perf_counter()
print(f"Elapsed time: {end - start}")
