import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import torch


HISTORY_LEN = 24
STEP_MINUTES = 5
N_SENSORS = 207

# These are computed exactly as in the supplied notebook:
# mean/std are calculated from the first 70% of METR-LA.
TRAIN_MEAN = 54.405914306640625
TRAIN_STD = 19.494253158569336


def load_speed_h5(path: str) -> Tuple[np.ndarray, list]:
    df = pd.read_hdf(path)
    df = df.sort_index()

    sensor_ids = [
        str(int(c)) if str(c).replace(".", "", 1).isdigit() else str(c)
        for c in df.columns
    ]

    speed = df.values.astype(np.float32)

    if speed.shape[1] != N_SENSORS:
        raise ValueError(
            f"Expected {N_SENSORS} sensors, got {speed.shape[1]}."
        )

    return speed, sensor_ids


def load_adjacency_pickle(path: str) -> Tuple[list, dict, np.ndarray]:
    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")

    if not isinstance(data, (tuple, list)) or len(data) != 3:
        raise ValueError(
            "Expected adjacency pickle as "
            "(sensor_ids, sensor_id_to_ind, adj_matrix)."
        )

    sensor_ids, sensor_id_to_ind, adj = data
    return sensor_ids, sensor_id_to_ind, np.asarray(adj, dtype=np.float32)


def build_time_features(start_row: int, history_len: int = HISTORY_LEN):
    t = np.arange(
        start_row,
        start_row + history_len,
    )

    minute_of_day = (t * STEP_MINUTES) % 1440
    day_of_week = ((t * STEP_MINUTES) // 1440) % 7

    tod_sin = np.sin(
        2 * np.pi * minute_of_day / 1440
    ).astype(np.float32)

    tod_cos = np.cos(
        2 * np.pi * minute_of_day / 1440
    ).astype(np.float32)

    dow_sin = np.sin(
        2 * np.pi * day_of_week / 7
    ).astype(np.float32)

    dow_cos = np.cos(
        2 * np.pi * day_of_week / 7
    ).astype(np.float32)

    return np.stack(
        [tod_sin, tod_cos, dow_sin, dow_cos],
        axis=-1,
    )


def make_model_inputs(history_raw: np.ndarray, start_row: int):
    history_raw = np.asarray(
        history_raw,
        dtype=np.float32,
    )

    if history_raw.shape != (HISTORY_LEN, N_SENSORS):
        raise ValueError(
            f"Expected history shape "
            f"({HISTORY_LEN}, {N_SENSORS}), "
            f"got {history_raw.shape}."
        )

    history_norm = (
        (history_raw - TRAIN_MEAN) / TRAIN_STD
    )

    time_feats = build_time_features(
        start_row,
        HISTORY_LEN,
    )

    x = torch.from_numpy(
        history_norm
    ).float().unsqueeze(0)

    tf = torch.from_numpy(
        time_feats
    ).float().unsqueeze(0)

    return x, tf


def make_adj_bias(adj_matrix: np.ndarray):
    eps = 1e-4
    return torch.log(
        torch.from_numpy(adj_matrix).float() + eps
    )
