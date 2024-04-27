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
from utils import df_from_listobject

SIMULATED_YEARS = 100000

# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = client.Dispatch("Excel.Application")

try:
    wb_path = sys.argv[1]
    wb = excel.Workbooks.Open(wb_path)
except IndexError:
    print(Path.cwd())
    wb = excel.Workbooks.Open(f"{Path.cwd()}/excelfile.xlsm")

# --------------------------------------
# Step 2: Import the existing DB records
# --------------------------------------

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
df_layer = df_from_listobject(wb.Worksheets("Database").ListObjects("Layer"))
df_layer = df_layer.drop(columns=["id"])
df_layer.to_sql(
    name="layer",
    con=engine,
    if_exists="append",
    index=False,
)

df_layerreinstatement = df_from_listobject(
    wb.Worksheets("Database").ListObjects("LayerReinstatement")
)
df_layerreinstatement = df_layerreinstatement.drop(columns=["id"])
df_layerreinstatement.to_sql(
    name="layerreinstatement",
    con=engine,
    if_exists="append",
    index=False,
)

df_modelfile = df_from_listobject(wb.Worksheets("Database").ListObjects("ModelFile"))
df_modelfile = df_modelfile.drop(columns=["id"])
df_modelfile.to_sql(
    name="modelfile",
    con=engine,
    if_exists="append",
    index=False,
)

df_modelyearloss = df_from_listobject(
    wb.Worksheets("Database").ListObjects("ModelYearLoss")
)
df_modelyearloss = df_modelyearloss.drop(columns=["id"])
df_modelyearloss.to_sql(
    name="modelyearloss",
    con=engine,
    if_exists="append",
    index=False,
)

# --------------------------------------
# Step 3: Read the input data
# --------------------------------------

# Read the input data
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
print(f"Elapsed time: {end - start}")

# --------------------------------------
# Step 5: Write the output data
# --------------------------------------

# Define the output worksheet and table
ws_output = wb.Worksheets("Output")

for DbModel in [LayerYearLoss, ResultLayerStatisticLoss]:
    query = select(DbModel)
    df_output = pd.read_sql(query, engine)
    table_output = ws_output.ListObjects(DbModel.__name__)

    # Clear the output table
    if table_output.DataBodyRange is None:
        pass
    else:
        table_output.DataBodyRange.Delete()

    # Define the range for writing the output data, then write
    cell_start = table_output.Range.Cells(2, 1)
    cell_end = table_output.Range.Cells(2, 1).Offset(
        len(df_output), len(df_output.columns)
    )
    ws_output.Range(cell_start, cell_end).Value = df_output.values

ws_output.Select()
win32api.MessageBox(0, "Done", "Python")

"""

ws_output.Select()

Other useful commands:

f = open('new_text_file.txt', 'x'
f.write('something')
f.close()
wb.Save()
wb.SaveAs('excelfile.xlsx')
wb.Close()
excel.Quit()
excel = None

"""
