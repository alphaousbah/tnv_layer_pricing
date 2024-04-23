# https://docs.sqlalchemy.org/en/20/orm/quickstart.html
# https://raaviblog.com/python-2-7-read-and-write-excel-file-with-win32com/
# https://learn.microsoft.com/en-us/dotnet/api/microsoft.office.interop.excel.listobject?view=excel-pia

from time import perf_counter

import win32api
from sqlalchemy.orm import Session
from win32com import client

from db import (
    Layer,
    LayerReinstatement,
    ModelFile,
    ModelYearLoss,
    ResultInstance,
    ResultLayer,
    engine,
)
from engine.function_layeryearloss import get_df_layeryearloss
from utils import df_from_listobject

# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = client.Dispatch("Excel.Application")

# wb_path = sys.argv[1]
# wb = excel.Workbooks.Open(wb_path)
wb = excel.Workbooks.Open(
    r"C:\Users\USER\Desktop\my-web-apps\layer_pricing\excelfile.xlsm"
)

# --------------------------------------
# Step 2: Import the existing DB records
# --------------------------------------

df_layer = df_from_listobject(wb.Worksheets("Database").ListObjects("Layer"))
df_layerreinstatement = df_from_listobject(
    wb.Worksheets("Database").ListObjects("LayerReinstatement")
)
df_modelfile = df_from_listobject(wb.Worksheets("Database").ListObjects("ModelFile"))
df_modelyearloss = df_from_listobject(
    wb.Worksheets("Database").ListObjects("ModelYearLoss")
)

with Session(engine) as session:
    for _, row in df_layer.iterrows():
        layer = Layer(
            occ_limit=row["occ_limit"],
            occ_deduct=row["occ_deduct"],
            agg_limit=row["agg_limit"],
            agg_deduct=row["agg_deduct"],
        )
        session.add(layer)
        # session.flush()
    for _, row in df_layerreinstatement.iterrows():
        layerreinstatement = LayerReinstatement(
            order=row["order"],
            number=row["number"],
            rate=row["rate"],
            layer_id=row["layer_id"],
        )
        session.add(layerreinstatement)
    for _, row in df_modelfile.iterrows():
        modelfile = ModelFile(years_simulated=row["years_simulated"])
        session.add(modelfile)
        # session.flush()
    for _, row in df_modelyearloss.iterrows():
        modelyearloss = ModelYearLoss(
            year=row["year"],
            day=row["day"],
            loss=row["loss"],
            loss_type=row["loss_type"],
            modelfile_id=row["modelfile_id"],
        )
        session.add(modelyearloss)
    session.commit()

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

    # Create the new relationships between layers and modelfiles
    for _, row in df_layer_modelfile.iterrows():
        layer = session.get(Layer, row["layer_id"])
        modelfile = session.get(ModelFile, row["modelfile_id"])
        layer.modelfiles.append(modelfile)

    # Calculate and save the layeryearlosses
    for layer_id in layer_ids:
        modelfiles_ids = [modelfile.id for modelfile in layer.modelfiles]
        print(modelfiles_ids)
        df = get_df_layeryearloss(layer_id, modelfiles_ids)

        # TODO: Save

    # Create a resultinstance
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
        )
        resultinstance.layers.append(resultlayer)

        # Create the relationships between resultlayers and modelfiles
        for modelfile in layer.modelfiles:
            resultlayer.modelfiles.append(modelfile)

end = perf_counter()
print(f"Elapsed time: {end - start}")

# --------------------------------------
# Step 4: Save in the database
# --------------------------------------


# --------------------------------------
# Step 4: Write the output data
# --------------------------------------

# Define the output worksheet and table
ws_output = wb.Worksheets("Output")
table_output = ws_output.ListObjects("LayerYearLoss")

# Clear the output table
if table_output.DataBodyRange is None:
    pass
else:
    table_output.DataBodyRange.Delete()


# Define the range for writing the output data, then write
cell_start = table_output.Range.Cells(2, 1)
cell_end = table_output.Range.Cells(2, 1).Offset(len(df), len(df.columns))
ws_output.Range(cell_start, cell_end).Value = df.values

ws_output.Select()
win32api.MessageBox(0, "Done", "Python")

"""

ws_output.Select()

Other useful commands:

f = open('new_text_file.txt', 'x'
f.write('something')
f.close()
wb.Save()
wb.SaveAs('updatedSample.xlsx')
wb.Close()
excel.Quit()
excel = None

"""
