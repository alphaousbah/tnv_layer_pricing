from typing import Type

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
from sqlalchemy.sql.selectable import Select
from win32com.client import CDispatch

from database import Base


def df_from_listobject(listobject: CDispatch) -> pd.DataFrame:
    """
    Converts a ListObject to a pandas DataFrame.

    This function extracts data from a ListObject's DataBodyRange and HeaderRowRange,
    and uses them to create a pandas DataFrame.

    :param listobject: The ListObject to convert.
    :return: The resulting DataFrame.
    """
    data = listobject.DataBodyRange() if listobject.DataBodyRange else []
    columns = listobject.HeaderRowRange()
    return pd.DataFrame(data, columns=columns[0])


def read_from_listobject_and_save(
    session: Session, worksheet: CDispatch, listobject_names: list[str]
) -> None:
    """
    Reads data from list objects in a database, processes it, and saves it to an SQL database.

    This function iterates over a list of listobjects, converts each object to a pandas DataFrame,
    drops the 'id' column from the DataFrame, and saves the DataFrame to an SQL database using
    the SQLAlchemy engine.

    :param session: The SQLAlchemy Session for database operations.
    :param worksheet: The Excel Worksheet containing the Excel ListObjects.
    :param listobject_names: A list of Excel ListObjects to be read from the Worksheet.
    :return: None
    """
    for listobject_name in listobject_names:
        df = df_from_listobject(worksheet.ListObjects(listobject_name))
        df = df.drop(columns=["id"])
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
        df.to_sql(
            name=str(listobject_name).lower(),
            con=session.connection(),
            if_exists="append",
            index=False,
        )
    return None


def write_df_in_listobjects(
    session: Session, DbModels: list[Type[Base]], ws_output: CDispatch
) -> None:
    """
    Write data from database models to corresponding ListObjects in an Excel worksheet.

    This function retrieves data from the specified database models, clears the existing data
    in the corresponding ListObjects in the Excel worksheet, and writes the new data to them.

    :param session: SQLAlchemy session for database access.
    :param DbModels: List of database model classes to retrieve data from.
    :param ws_output: Excel worksheet object where data will be written.
    :return: None
    """
    for DbModel in DbModels:
        query = select(DbModel)
        df_output = pd.read_sql(query, session.connection())
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


def get_single_result(session: Session, query: Select, item_name: str):
    try:
        return session.execute(query).scalar_one()
    except NoResultFound:
        raise ValueError(f"{item_name} not found")
    except MultipleResultsFound:
        raise ValueError(f"Multiple {item_name} found")


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
