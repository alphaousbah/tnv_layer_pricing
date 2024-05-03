# https://docs.sqlalchemy.org/en/20/orm/quickstart.html
# https://raaviblog.com/python-2-7-read-and-write-excel-file-with-win32com/
# https://learn.microsoft.com/en-us/dotnet/api/microsoft.office.interop.excel.listobject?view=excel-pia
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import win32api
from sqlalchemy import select
from sqlalchemy.orm import Session
from win32com import client

from db import (
    Layer,
    LayerYearLoss,
    ModelFile,
    ResultInstance,
    ResultLayer,
    ResultLayerReinstatement,
    ResultLayerStatisticLoss,
    engine,
)
from engine.function_layeryearloss import get_df_layeryearloss
from engine.function_resultlayerstatisticloss import get_df_resultlayerstatisticloss
from utils import (
    df_from_listobject,
    read_from_listobject_and_save,
    write_df_in_listobjects,
)

SIMULATED_YEARS = 100_000

# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = client.Dispatch("Excel.Application")

try:
    wb_path = sys.argv[1]
    wb = excel.Workbooks.Open(wb_path)
except IndexError:
    wb = excel.Workbooks.Open(f"{Path.cwd()}/run_pricing.xlsm")

# --------------------------------------
# Step 2: Import the existing DB records
# --------------------------------------

read_from_listobject_and_save(
    ws_database=wb.Worksheets("Database"),
    listobjects=[
        "Layer",
        "LayerReinstatement",
        "ModelFile",
        "ModelYearLoss",
    ],
    engine=engine,
)


# --------------------------------------
# Step 3: Read the input data
# --------------------------------------

df_layer_modelfile = df_from_listobject(
    wb.Worksheets("Input").ListObjects("layer_modelfile")
)

# --------------------------------------
# Step 4: Process
# --------------------------------------

start = perf_counter()
layer_ids = df_layer_modelfile["layer_id"].unique()

with Session(engine) as session:
    # Delete the previous relationships between layers and modelfiles
    # TODO: Correct the code above by retrieving all the analysis layers
    for layer_id in layer_ids:
        layer = session.get(Layer, layer_id)
        layer.modelfiles.clear()

    # Create and save the new relationships between layers and modelfiles
    for _, row in df_layer_modelfile.iterrows():
        layer = session.get(Layer, row["layer_id"])
        modelfile = session.get(ModelFile, row["modelfile_id"])
        layer.modelfiles.append(modelfile)

    # Calculate and save the layeryearlosses
    for layer_id in layer_ids:
        modelfiles_ids = [modelfile.id for modelfile in layer.modelfiles]
        df_layeryearloss = get_df_layeryearloss(
            layer_id, modelfiles_ids, SIMULATED_YEARS
        )

        df_layeryearloss.to_sql(
            name="layeryearloss",
            con=engine,
            if_exists="append",
            index=False,
        )

    # Create and save the resultinstance
    resultinstance = ResultInstance(name="Run 1")
    session.add(resultinstance)

    # Create the resultlayers
    for layer_id in layer_ids:
        layer = session.get(Layer, layer_id)
        resultlayer = ResultLayer(
            occ_limit=layer.occ_limit,
            occ_deduct=layer.occ_deduct,
            agg_limit=layer.agg_limit,
            agg_deduct=layer.agg_deduct,
            source_id=layer.id,
        )
        resultinstance.layers.append(resultlayer)

        # Create and save the resultlayerreinstatements
        for layerreinstatement in layer.reinstatements:
            resultlayerreinstatement = ResultLayerReinstatement(
                order=layerreinstatement.order,
                number=layerreinstatement.number,
                rate=layerreinstatement.rate,
            )
            resultlayer.reinstatements.append(resultlayerreinstatement)

        # Create and save the relationships between resultlayers and modelfiles
        for modelfile in layer.modelfiles:
            resultlayer.modelfiles.append(modelfile)

        # Calculate and save the resultlayerstatisticlosses
        df_resultlayerstatisticloss = get_df_resultlayerstatisticloss(
            resultlayer.source_id, SIMULATED_YEARS
        )
        df_resultlayerstatisticloss.to_sql(
            name="resultlayerstatisticloss",
            con=engine,
            if_exists="append",
            index=False,
        )
    session.commit()

end = perf_counter()
print(f"Calculation time: {end - start}")

# --------------------------------------
# Step 5: Write the output data
# --------------------------------------

start = perf_counter()
ws_output = wb.Worksheets("Output")
write_df_in_listobjects(
    DbModels=[LayerYearLoss, ResultLayerStatisticLoss],
    ws_output=ws_output,
    engine=engine,
)
ws_output.Select()
end = perf_counter()
print(f"Writing in Excel time: {end - start}")
