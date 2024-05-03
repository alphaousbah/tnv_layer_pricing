import pandas as pd
from sqlalchemy import select


def df_from_listobject(listobject):
    if listobject.DataBodyRange is None:
        data = []
    else:
        data = listobject.DataBodyRange()
    columns = listobject.HeaderRowRange()
    return pd.DataFrame(data, columns=columns[0])


def read_from_listobject_and_save(ws_database, listobjects, engine):
    for listobject in listobjects:
        df = df_from_listobject(ws_database.ListObjects(listobject))
        df = df.drop(columns=["id"])
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
        df.to_sql(
            name=str(listobject).lower(),
            con=engine,
            if_exists="append",
            index=False,
        )
    return None


def write_df_in_listobjects(DbModels, ws_output, engine):
    for DbModel in DbModels:
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
    return None


"""

win32api.MessageBox(0, "Done", "Python")
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
