import numpy as np
import pandas as pd
import xarray as xr


def read_forcing_at_sites(path : str, avg=True) -> pd.DataFrame:

    ds = xr.open_dataset(path)

    dim_map = {}
    if 'valid_time' in ds.dims: dim_map['valid_time'] = 'time'
    if 'latitude' in ds.dims: dim_map['latitude'] = 'lat'
    if 'longitude' in ds.dims: dim_map['longitude'] = 'lon'
    ds = ds.rename(dim_map)

    ds = ds.assign_coords(
        lat=np.round(ds['lat'].values.astype(np.float64), 1),
        lon=np.round(ds['lon'].values.astype(np.float64), 1),
    )

    w1, w2, w3 = 7.0 / 30.0, 21.0 / 30.0, 2.0 / 30.0
    tsoil_full = (ds['stl1'] * w1 + ds['stl2'] * w2 + ds['stl3'] * w3)
    wsoil_full = (ds['swvl1'] * w1 + ds['swvl2'] * w2 + ds['swvl3'] * w3)

    # Объединим в удобный временный датасет
    tmp = xr.Dataset({"Tsoil": tsoil_full, "Wsoil": wsoil_full})

    stk = tmp.stack(pt=('lat', 'lon'))  # dims: time, pt
    pt_index = stk.indexes['pt']  # pandas.MultiIndex

    obs_coordinates = pd.read_csv('data/obs_coordinates.csv', header=None).to_numpy()

    # Сопоставим пары к позициям индекса (без KeyError)
    pairs = list(map(tuple, obs_coordinates))
    pos = pt_index.get_indexer(pairs)  # -1 если пары нет (на всякий случай)
    pos = pos[pos >= 0]

    # Выбираем только нужные точки и разворачиваем обратно в (time, lat, lon)
    sub = stk.isel(pt=np.unique(pos))  # time × selected-pt
    sub = sub.unstack('pt').transpose('time', 'lat', 'lon')

    forcing = (
        sub[['Tsoil', 'Wsoil']]
        .to_dataframe()
        .reset_index()
        .loc[:, ['lat', 'lon', 'time', 'Tsoil', 'Wsoil']]
    )
    forcing = forcing.sort_values(['lat', 'lon', 'time'])

    if avg:
        forcing = forcing.groupby(['lat', 'lon'], as_index=False)[['Tsoil', 'Wsoil']].mean()
    return forcing
