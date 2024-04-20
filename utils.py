import pandas as pd


def df_from_listobject(listobject):
    if listobject.DataBodyRange is None:
        data = []
    else:
        data = listobject.DataBodyRange()
    columns = listobject.HeaderRowRange()

    return pd.DataFrame(data, columns=columns[0])
