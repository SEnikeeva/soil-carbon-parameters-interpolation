import os
from typing import List

import numpy as np
import pandas as pd

base_path = '../data'
EXCEL_PATH = os.path.join(base_path, '3. Данные 18092025.xlsx')

def find_header_row(df):
    for i in range(min(10, len(df))):
        row = df.iloc[i].tolist()

        if ("N" in row) and ("E" in row):
            return i
    return None

def clean_headers(headers) -> List[str]:
    out = []
    for i, h in enumerate(headers):
        if isinstance(h, str) and h.strip():
            out.append(h.strip())
        else:
            out.append(f"col{i}")
    return out

def parse_sheet_side_by_side(xls_path, sheet_name):
    """Лист имеет две «секции» в СТОЛБЦАХ:
       слева Растения (N,E, ...), справа Почвы (N,E, ...)
       Берем пары (N,E) слева как растения, вторую пару (N,E) справа как почвы.
    """
    df = pd.read_excel(xls_path, sheet_name=sheet_name, header=None)
    hdr_row = find_header_row(df)
    if hdr_row is None:
        return pd.DataFrame(), pd.DataFrame()

    headers = clean_headers(df.iloc[hdr_row].tolist())
    data = df.iloc[hdr_row+1:].copy()

    if data.shape[1] > len(headers):
        data = data.iloc[:, :len(headers)].copy()
    else:
        headers = headers[:data.shape[1]]

    data.columns = headers

    prod_col = next((c for c in data.columns if isinstance(c, str) and "NPP" in c), None)
    soil_col = next((c for c in data.columns if isinstance(c, str) and "слое" in c), None)
    cm_col = next((c for c in data.columns if isinstance(c, str) and "Сm" in c), None)
    fraction_col = next((c for c in data.columns if isinstance(c, str) and "фракции" in c), None)
    bd_col = next((c for c in data.columns if isinstance(c, str) and "Плотность" in c), None)

    # Найдем позиции столбцов 'N' и 'E'
    n_pos = [i for i, h in enumerate(headers) if h == "N"]
    e_pos = [i for i, h in enumerate(headers) if h == "E"]
    pairs = []
    used_e = set()
    for ni in n_pos:
        ei = ni + 1 if (ni + 1) in e_pos else None
        if ei is None:
            rights = [e for e in e_pos if e > ni and e not in used_e]
            if rights:
                ei = rights[0]
        if ei is not None:
            pairs.append((ni, ei))
            used_e.add(ei)

    veg_df = pd.DataFrame(); soil_df = pd.DataFrame()
    if len(pairs) >= 1:
        ni, ei = pairs[0]
        veg_df = data.iloc[:, [ni, ei]].copy()
        veg_df.columns = ["N", "E"]
        if prod_col and prod_col in data.columns:
            prod = pd.to_numeric(data[prod_col], errors="coerce")
            veg_df["organic"] = prod
            veg_df["organic"] = prod / 12.0
        veg_df["group"] = "veg"
        veg_df["province"] = sheet_name
        veg_df["veg_type"] = data["Veg_type_veg"]
    if len(pairs) >= 2:
        ni, ei = pairs[1]
        soil_df = data.iloc[:, [ni, ei]].copy()
        soil_df.columns = ["N", "E"]
        if soil_col and soil_col in data.columns:
            soil = pd.to_numeric(data[soil_col], errors="coerce")
            soil_df["c"] = soil
            # soil_df["c"] = soil / 10.
        if cm_col and cm_col in data.columns:
            cm = pd.to_numeric(data[cm_col], errors="coerce")
            soil_df["Cm"] = cm
        if fraction_col and fraction_col in data.columns:
            fraction = pd.to_numeric(data[fraction_col], errors="coerce")
            soil_df["fraction"] = fraction
        if bd_col and bd_col in data.columns:
            bd = pd.to_numeric(data[bd_col], errors="coerce")
            soil_df["bd"] = bd
        soil_df["veg_type"] = data["Veg_type_soil"]
        soil_df["group"] = "soil"
        soil_df["province"] = sheet_name

    for d in (veg_df, soil_df):
        if not d.empty:
            d["N"] = pd.to_numeric(d["N"], errors="coerce").apply(lambda x: np.round(x, 1))
            d["E"] = pd.to_numeric(d["E"], errors="coerce").apply(lambda x: np.round(x, 1))
            d.dropna(subset=["N", "E"], inplace=True)
    return veg_df, soil_df


def read_prod_data(path: str = 'prod_c.csv', only_soil: bool = False) -> pd.DataFrame:

    # значение органического вещества определяется для каждой почвенной точки
    # как среднее значение по провинции

    prod_data = pd.read_csv(path)
    prod_data = prod_data.sort_values(by=['lat', 'lon']).reset_index(drop=True)

    veg_means = (
        prod_data[prod_data["group"] == "veg"]
        .groupby("province")["organic"]
        .mean()
    )

    prod_data.loc[prod_data["group"] == "soil", "organic"] = (
        prod_data.loc[prod_data["group"] == "soil", "province"].map(veg_means)
    )

    if only_soil:
        return prod_data[prod_data["group"] == "soil"].reset_index(drop=True)
    else:
        return prod_data

if __name__ == '__main__':
    # --- Читаем книгу и парсим все целевые листы ---
    xls = pd.ExcelFile(EXCEL_PATH)
    veg_list, soil_list = [], []
    for sh in xls.sheet_names[:-1]:
        v, s = parse_sheet_side_by_side(EXCEL_PATH, sh)
        if not v.empty: veg_list.append(v)
        if not s.empty: soil_list.append(s)

    veg_df = pd.concat(veg_list, ignore_index=True) if veg_list else pd.DataFrame(
        columns=["N", "E", "group", "province", "veg_type"])
    soil_df = pd.concat(soil_list, ignore_index=True) if soil_list else pd.DataFrame(
        columns=["N", "E", "group", "province", "veg_type"])
    soil_df = soil_df[~soil_df.duplicated(subset=["N", "E"], keep='last')].reset_index(drop=True)

    combined = pd.concat([veg_df, soil_df], ignore_index=True)
    combined = combined.rename(columns={'N': 'lat', 'E': 'lon'})

    combined.to_csv(os.path.join(base_path, 'prod_c.csv'), index=False)
