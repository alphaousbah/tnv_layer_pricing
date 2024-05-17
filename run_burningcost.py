# https://docs.sqlalchemy.org/en/20/orm/quickstart.html
# https://raaviblog.com/python-2-7-read-and-write-excel-file-with-win32com/
# https://learn.microsoft.com/en-us/dotnet/api/microsoft.office.interop.excel.listobject?view=excel-pia
import sys
from pathlib import Path
from time import perf_counter

from sqlalchemy.orm import Session
from win32com import client

from database import (
    Analysis,
    HistoLossFile,
    Layer,
    LayerBurningCost,
    PremiumFile,
    engine,
)
from engine.function_burningcost import get_df_burningcost
from utils import (
    df_from_listobject,
    read_from_listobject_and_save,
    write_df_in_listobjects,
)

# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = client.Dispatch("Excel.Application")

try:
    wb_path = sys.argv[1]
    wb = excel.Workbooks.Open(wb_path)
except IndexError:
    wb = excel.Workbooks.Open(f"{Path.cwd()}/run_burningcost.xlsm")

# --------------------------------------
# Step 2: Import the existing DB records
# --------------------------------------

read_from_listobject_and_save(
    ws_database=wb.Worksheets("Database"),
    listobjects=[
        "Analysis",
        "Layer",
        "LayerReinstatement",
        "PremiumFile",
        "Premium",
        "HistoLossFile",
        "HistoLoss",
    ],
    engine=engine,
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

# Delete the previous relationships between layers and premiumfiles/histolossfiles
# Then create and save the new relationships between layers and premiumfiles/histolossfiles
with Session(engine) as session:
    analysis = session.get(Analysis, analysis_id)

    for layer in analysis.layers:
        layer.premiumfiles.clear()
        layer.histolossfiles.clear()

    for _, row in df_layer_premiumfile.iterrows():
        layer = session.get(Layer, row["layer_id"])
        premiumfile = session.get(PremiumFile, row["premiumfile_id"])
        layer.premiumfiles.append(premiumfile)

    for _, row in df_layer_histolossfile.iterrows():
        layer = session.get(Layer, row["layer_id"])
        histolossfile = session.get(HistoLossFile, row["histolossfile_id"])
        layer.histolossfiles.append(histolossfile)

    session.commit()

    # Get df_burningcost
    df_burningcost = get_df_burningcost(analysis_id, start_year, end_year, session)

# --------------------------------------
# Step 5: Write the output data
# --------------------------------------

# Define the output worksheet and table
ws_output = wb.Worksheets("Output")
write_df_in_listobjects(
    DbModels=[LayerBurningCost],
    ws_output=ws_output,
    engine=engine,
)
ws_output.Select()
end = perf_counter()
print(f"Elapsed time: {end - start}")