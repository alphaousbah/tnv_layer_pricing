from time import perf_counter

import numpy as np
import pandas as pd
from numba import njit

df = pd.DataFrame({"a": range(5)})

df.loc[0, "b"] = df.loc[0, "a"]
df.loc[0, "c"] = df.loc[0, "a"]

start = perf_counter()
for i in range(1, len(df)):
    df.loc[i, "b"] = df.loc[i - 1, "b"] + 10 * df.loc[i, "a"]
    df.loc[i, "c"] = df.loc[i - 1, "c"] + 10 * df.loc[i, "a"]

end = perf_counter()
print(df)
print(f"Classical loop: {end - start}")


@njit
def reinst(x, y, a):
    print(type(x))
    print(type(y))
    records_count = a.shape[0]
    b = np.empty(records_count)
    c = np.empty(records_count)
    b[0] = a[0] + x
    c[0] = a[0] + y
    for i in range(1, records_count):
        b[i] = b[i - 1] + 10 * a[i]
        c[i] = c[i - 1] + 10 * a[i]
    return b, c


start = perf_counter()
df["b"], df["c"] = reinst(10, 20, df["a"].values.T)
end = perf_counter()
print(df)
print(f"Numba loop: {end - start}")
