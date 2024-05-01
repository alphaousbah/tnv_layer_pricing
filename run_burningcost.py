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
    engine,
)

from utils import df_from_listobject


# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = client.Dispatch("Excel.Application")

try:
    wb_path = sys.argv[1]
    wb = excel.Workbooks.Open(wb_path)
except IndexError:
    print(Path.cwd())
    wb = excel.Workbooks.Open(f"{Path.cwd()}/run_burningcost.xlsm")

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
