import gc
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

DEFAULT_RENAME_MAP = {
    "LON": "lon",
    "LAT": "lat",
    "# Timestamp": "time",
    "h3_cell_10": "h3",
    "TRIP": "trip_id",
}

TRIP_REQUIRED_COLUMNS = ("trip_id", "time", "lon", "lat", "h3")


def _build_trip_rename_map(
    h3_resolution: Optional[int] = None,
    rename_map: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    effective_map = dict(DEFAULT_RENAME_MAP if rename_map is None else rename_map)
    if rename_map is None and h3_resolution is not None:
        effective_map.pop("h3_cell_10", None)
        effective_map[f"h3_cell_{h3_resolution}"] = "h3"
    return effective_map


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _ensure_parent_dir(path_like: str | Path) -> Path:
    path = _as_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_existing_path(path_like: str | Path) -> str:
    return _as_path(path_like).resolve().as_posix()


def parse_trip_times(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")

    parsed = pd.to_datetime(series, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    parsed = pd.Series(parsed, index=series.index, copy=False).astype("datetime64[ns]")
    if parsed.isna().any():
        fallback = pd.to_datetime(series[parsed.isna()], errors="coerce")
        fallback = pd.Series(fallback, index=series[parsed.isna()].index, copy=False).astype(
            "datetime64[ns]"
        )
        parsed.loc[parsed.isna()] = fallback
    return parsed


def prepare_trip_df(
    trip_data: str | Path | pd.DataFrame,
    required_columns: Optional[Sequence[str]] = None,
    h3_resolution: Optional[int] = None,
    rename_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    if isinstance(trip_data, (str, Path)):
        trip_df = pd.read_csv(trip_data)
    else:
        trip_df = trip_data.copy()

    rename_map = {
        key: value
        for key, value in _build_trip_rename_map(h3_resolution, rename_map=rename_map).items()
        if key in trip_df.columns
    }
    if rename_map:
        trip_df = trip_df.rename(columns=rename_map)

    if "time" in trip_df.columns:
        trip_df["time"] = parse_trip_times(trip_df["time"])

    if required_columns is not None:
        missing_columns = [column for column in required_columns if column not in trip_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    return trip_df


def _select_trip_rows(trip_df: pd.DataFrame, trip_id: str) -> pd.DataFrame:
    trip_slice = trip_df[trip_df["trip_id"] == trip_id].copy()
    if len(trip_slice) == 0:
        raise ValueError(f"No rows found for trip_id={trip_id}")
    return trip_slice.sort_values("time").reset_index(drop=True)


def _interpolate_trip_coords(
    time_one: pd.Timestamp,
    lon_one: float,
    lat_one: float,
    time_two: pd.Timestamp,
    lon_two: float,
    lat_two: float,
    target_time: pd.Timestamp,
) -> tuple[float, float]:
    total_duration = (time_two - time_one).total_seconds()
    if total_duration == 0:
        return lon_one, lat_one
    elapsed = (target_time - time_one).total_seconds()
    ratio = max(0.0, min(1.0, elapsed / total_duration))
    return lon_one + ratio * (lon_two - lon_one), lat_one + ratio * (lat_two - lat_one)


def _find_known_trip_neighbors(
    trip_df: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> tuple[int, int]:
    previous_index = start_index - 1
    while previous_index >= 0 and pd.isna(trip_df.loc[previous_index, "lon"]):
        previous_index -= 1

    next_index = end_index + 1
    while next_index < len(trip_df) and pd.isna(trip_df.loc[next_index, "lon"]):
        next_index += 1

    return previous_index, next_index


def compute_longest_trip_duration_seconds(
    trip_data: str | Path | pd.DataFrame,
    rename_map: Optional[Mapping[str, str]] = None,
) -> float:
    trip_df = prepare_trip_df(
        trip_data,
        required_columns=("trip_id", "time"),
        rename_map=rename_map,
    )
    durations = (
        trip_df.groupby("trip_id")["time"].max() - trip_df.groupby("trip_id")["time"].min()
    ).dt.total_seconds()
    durations = durations.dropna()
    return float(durations.max()) if len(durations) else 0.0


def clean_trip_dataset(
    trip_data: str | Path | pd.DataFrame,
    mean_duration_seconds: Optional[float] = 13282.0,
    std_duration_seconds: Optional[float] = 4486.14,
    duration_tolerance_seconds: float = 1800.0,
    output_path: Optional[str | Path] = None,
    h3_resolution: Optional[int] = None,
    rename_map: Optional[Mapping[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trip_df = prepare_trip_df(
        trip_data,
        required_columns=TRIP_REQUIRED_COLUMNS,
        h3_resolution=h3_resolution,
        rename_map=rename_map,
    )

    durations = trip_df.groupby("trip_id")["time"].agg(
        lambda values: (values.max() - values.min()).total_seconds()
    )
    durations = durations.dropna()
    if durations.empty:
        raise ValueError("No valid trip durations were found in the dataset.")

    effective_mean = float(mean_duration_seconds) if mean_duration_seconds is not None else float(durations.mean())
    effective_std = float(std_duration_seconds) if std_duration_seconds is not None else float(durations.std(ddof=0))
    lower_bound = effective_mean - duration_tolerance_seconds
    upper_bound = effective_mean + duration_tolerance_seconds

    valid_trip_ids = durations[(durations >= lower_bound) & (durations <= upper_bound)].index
    outlier_trip_ids = durations[(durations < lower_bound) | (durations > upper_bound)].index

    trips_clean = trip_df[trip_df["trip_id"].isin(valid_trip_ids)].copy()
    trips_clean = trips_clean[["trip_id", "time", "lon", "lat", "h3"]]

    if output_path is not None:
        output_file = _ensure_parent_dir(output_path)
        trips_clean.to_csv(output_file, index=False)

    stats = {
        "mean_duration_seconds": effective_mean,
        "std_duration_seconds": effective_std,
        "lower_bound_seconds": lower_bound,
        "upper_bound_seconds": upper_bound,
        "original_trip_count": int(trip_df["trip_id"].nunique()),
        "cleaned_trip_count": int(trips_clean["trip_id"].nunique()),
        "outlier_trip_count": int(len(outlier_trip_ids)),
    }
    return trips_clean, stats


def augment_trip(
    trip_data: str | Path | pd.DataFrame,
    trip_id: str,
    longest_trip_duration_seconds: Optional[float] = None,
    h3_resolution: int = 10,
    n_bins: int = 128,
    verbose: bool = True,
    rename_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    trip_df = prepare_trip_df(
        trip_data,
        required_columns=TRIP_REQUIRED_COLUMNS,
        h3_resolution=h3_resolution,
        rename_map=rename_map,
    )

    if longest_trip_duration_seconds is None:
        longest_trip_duration_seconds = compute_longest_trip_duration_seconds(trip_df)
        if verbose:
            print(f"Computed longest_trip_duration_seconds={longest_trip_duration_seconds} seconds")

    if longest_trip_duration_seconds <= 0:
        raise ValueError("longest_trip_duration_seconds must be positive.")

    trip_filtered = _select_trip_rows(trip_df, trip_id).loc[:, ["time", "lon", "lat", "h3", "trip_id"]]

    trip_start = trip_filtered["time"].iloc[0]
    trip_end = trip_start + pd.Timedelta(seconds=longest_trip_duration_seconds)
    bin_edges = pd.date_range(start=trip_start, end=trip_end, periods=n_bins + 1)
    bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    trip_filtered["seconds_from_start"] = (
        trip_filtered["time"] - trip_start
    ).dt.total_seconds().clip(lower=0, upper=longest_trip_duration_seconds)
    trip_filtered["bin_idx"] = (
        trip_filtered["seconds_from_start"] / longest_trip_duration_seconds * n_bins
    ).astype(int).clip(0, n_bins - 1)

    bins_with_data: dict[int, dict[str, Any]] = {}
    times = trip_filtered["time"]
    for bin_index in range(n_bins):
        start_edge = bin_edges[bin_index]
        end_edge = bin_edges[bin_index + 1]
        in_bin = trip_filtered[(times >= start_edge) & (times < end_edge)]
        if len(in_bin) == 0:
            continue
        center = bin_centers[bin_index]
        closest_index = (in_bin["time"] - center).abs().idxmin()
        row = trip_filtered.loc[closest_index]
        bins_with_data[bin_index] = {
            "time": row["time"],
            "lon": row["lon"],
            "lat": row["lat"],
            "h3": row["h3"],
            "trip_id": row["trip_id"],
            "bin_idx": bin_index,
        }

    known_bins = sorted(bins_with_data.keys())
    if len(known_bins) == 0:
        raise ValueError("No bins populated from original data; cannot augment.")
    last_known_row = bins_with_data[known_bins[-1]]

    def find_prev_next(index: int) -> tuple[Optional[int], Optional[int]]:
        previous_bin = max([value for value in known_bins if value < index], default=None)
        next_bin = min([value for value in known_bins if value > index], default=None)
        return previous_bin, next_bin

    new_rows: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        if bin_index in bins_with_data:
            continue

        previous_bin, next_bin = find_prev_next(bin_index)

        if previous_bin is None and next_bin is None:
            reference_row = trip_filtered.iloc[0]
            lon_value = reference_row["lon"]
            lat_value = reference_row["lat"]
            h3_value = reference_row["h3"]
            time_value = bin_centers[bin_index]
        elif previous_bin is None:
            reference_row = bins_with_data[next_bin]
            lon_value = reference_row["lon"]
            lat_value = reference_row["lat"]
            h3_value = reference_row["h3"]
            time_value = reference_row["time"] - pd.Timedelta(
                seconds=(next_bin - bin_index) * longest_trip_duration_seconds / n_bins
            )
        elif next_bin is None:
            reference_row = last_known_row
            lon_value = reference_row["lon"]
            lat_value = reference_row["lat"]
            h3_value = reference_row["h3"]
            time_value = reference_row["time"] + pd.Timedelta(
                seconds=(bin_index - known_bins[-1]) * longest_trip_duration_seconds / n_bins
            )
        else:
            previous_row = bins_with_data[previous_bin]
            next_row = bins_with_data[next_bin]
            gap = next_bin - previous_bin
            ratio = (bin_index - previous_bin) / gap
            time_value = previous_row["time"] + (next_row["time"] - previous_row["time"]) * ratio

            if previous_row["h3"] == next_row["h3"]:
                lon_value = (previous_row["lon"] + next_row["lon"]) / 2
                lat_value = (previous_row["lat"] + next_row["lat"]) / 2
                h3_value = previous_row["h3"]
            else:
                lon_value = previous_row["lon"] + ratio * (next_row["lon"] - previous_row["lon"])
                lat_value = previous_row["lat"] + ratio * (next_row["lat"] - previous_row["lat"])
                h3_value = h3.latlng_to_cell(lat_value, lon_value, h3_resolution)

        new_rows.append(
            {
                "time": time_value,
                "lon": lon_value,
                "lat": lat_value,
                "h3": h3_value,
                "trip_id": trip_id,
                "bin_idx": bin_index,
            }
        )

    trip_augmented = (
        pd.concat([trip_filtered, pd.DataFrame(new_rows)], ignore_index=True)
        .sort_values(["bin_idx", "time"], kind="mergesort")
        .reset_index(drop=True)
    )

    filled_bins = set(trip_augmented["bin_idx"].unique())
    missing_after = sorted(set(range(n_bins)) - filled_bins)
    trip_augmented = trip_augmented.drop(columns=["seconds_from_start", "bin_idx"])

    if verbose:
        print(
            f"Original rows: {len(trip_filtered)} | Added rows: {len(new_rows)} | Total: {len(trip_augmented)}"
        )
        print(
            f"Binned coverage: {len(filled_bins)} / {n_bins} bins filled; remaining missing bins: {len(missing_after)}"
        )

    return trip_augmented


def augment_all_trips(
    trip_data: str | Path | pd.DataFrame,
    longest_trip_duration_seconds: Optional[float] = None,
    h3_resolution: int = 10,
    n_bins: int = 128,
    verbose: bool = True,
    rename_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    trip_df = prepare_trip_df(
        trip_data,
        required_columns=TRIP_REQUIRED_COLUMNS,
        h3_resolution=h3_resolution,
        rename_map=rename_map,
    )

    if longest_trip_duration_seconds is None:
        longest_trip_duration_seconds = compute_longest_trip_duration_seconds(trip_df)
        if verbose:
            print(
                f"Computed longest_trip_duration_seconds={longest_trip_duration_seconds} seconds across all trips"
            )

    augmented_frames = []
    for trip_id in trip_df["trip_id"].dropna().unique():
        augmented_frames.append(
            augment_trip(
                trip_df,
                trip_id,
                longest_trip_duration_seconds=longest_trip_duration_seconds,
                h3_resolution=h3_resolution,
                n_bins=n_bins,
                verbose=verbose,
            )
        )
    return pd.concat(augmented_frames, ignore_index=True)


def generate_float32_bitpacked_colormap(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    h3_col: str = "h3",
    plot_graph: bool = False,
) -> dict[str, dict[str, Any]]:
    centroids = df.groupby(h3_col).agg(
        avg_lon=(lon_col, "mean"),
        avg_lat=(lat_col, "mean"),
    ).reset_index()

    if centroids.empty:
        return {}

    point_count = len(centroids)
    print("=" * 80)
    print("GENERATING BIT-PACKED FLOAT32 RGB COLORS")
    print("=" * 80)
    print(f"Points: {point_count:,}")

    lon_min = centroids["avg_lon"].min()
    lon_max = centroids["avg_lon"].max()
    lat_min = centroids["avg_lat"].min()
    lat_max = centroids["avg_lat"].max()

    lon_span = max(lon_max - lon_min, 1e-10)
    lat_span = max(lat_max - lat_min, 1e-10)

    normalized_lon = ((centroids["avg_lon"] - lon_min) / lon_span).values
    normalized_lat = ((centroids["avg_lat"] - lat_min) / lat_span).values

    print("\nGeographic bounds:")
    print(f"Longitude: {lon_min:.6f} to {lon_max:.6f}")
    print(f"Latitude:  {lat_min:.6f} to {lat_max:.6f}")

    main_bits = 18
    shared_bits = 5
    total_bits = main_bits + shared_bits
    max_value = (1 << total_bits) - 1

    print("\nBit packing strategy:")
    print(f"Main channel bits (R for lat, B for lon): {main_bits}")
    print(f"Shared G channel bits per coord: {shared_bits}")
    print(f"Total bits per coordinate: {total_bits}")
    print(f"Max unique values per axis: {max_value + 1:,}")
    print(f"Theoretical max unique colors: {(max_value + 1) ** 2:,}")

    lat_index = np.clip((normalized_lat * max_value).round().astype(np.int64), 0, max_value)
    lon_index = np.clip((normalized_lon * max_value).round().astype(np.int64), 0, max_value)

    centroids["lat_idx"] = lat_index
    centroids["lon_idx"] = lon_index
    collision_offset = centroids.groupby(["lat_idx", "lon_idx"]).cumcount().values
    lon_index = np.clip(lon_index + collision_offset, 0, max_value)

    red_int = (lat_index >> shared_bits) & ((1 << main_bits) - 1)
    blue_int = (lon_index >> shared_bits) & ((1 << main_bits) - 1)
    lat_low = lat_index & ((1 << shared_bits) - 1)
    lon_low = lon_index & ((1 << shared_bits) - 1)
    green_int = (lat_low << shared_bits) | lon_low

    red_max = (1 << main_bits) - 1
    green_max = (1 << (2 * shared_bits)) - 1
    blue_max = (1 << main_bits) - 1

    red = (1.0 - red_int / red_max).astype(np.float32)
    green = (1.0 - green_int / green_max).astype(np.float32)
    blue = (1.0 - blue_int / blue_max).astype(np.float32)

    red = 0.05 + red * 0.9
    green = 0.05 + green * 0.9
    blue = 0.05 + blue * 0.9

    print("\nChannel encoding (DARKENS with increasing lat/lon):")
    print(f"R: latitude  high {main_bits} bits (inverted) -> [{red.min():.3f}, {red.max():.3f}]")
    print(f"G: lat_low|lon_low (inverted) -> [{green.min():.3f}, {green.max():.3f}]")
    print(f"B: longitude high {main_bits} bits (inverted) -> [{blue.min():.3f}, {blue.max():.3f}]")

    red_f32 = red.astype(np.float32)
    green_f32 = green.astype(np.float32)
    blue_f32 = blue.astype(np.float32)

    centroids["color"] = list(zip(red_f32, green_f32, blue_f32))
    centroids["color_plot"] = list(zip(red_f32, green_f32, blue_f32))
    centroids["X"] = normalized_lon
    centroids["Y"] = normalized_lat
    centroids = centroids.set_index(h3_col)

    h3_dict = {
        h3_cell: {
            "position": (row["avg_lon"], row["avg_lat"]),
            "normalized_pos": (row["X"], row["Y"]),
            "color": row["color"],
            "color_plot": row["color_plot"],
        }
        for h3_cell, row in centroids.iterrows()
    }

    unique_colors = len(set(centroids["color"]))
    collision_rate = (point_count - unique_colors) / point_count * 100

    print("\nResults:")
    print(f"Total points: {point_count:,}")
    print(f"Unique colors: {unique_colors:,}")
    print(f"Uniqueness: {unique_colors / point_count * 100:.2f}%")
    if unique_colors < point_count:
        print(f"Collisions: {point_count - unique_colors:,} ({collision_rate:.2f}%)")
    else:
        print("All colors are unique.")

    print("\nDecoding verification (sample):")
    sample_size = min(5, point_count)
    for index in range(sample_size):
        original_lat = normalized_lat[index]
        original_lon = normalized_lon[index]
        decoded_lat = ((red_int[index] << shared_bits) | lat_low[index]) / max_value
        decoded_lon = ((blue_int[index] << shared_bits) | lon_low[index]) / max_value
        offset = collision_offset[index]
        offset_suffix = f" (offset={offset})" if offset > 0 else ""
        print(
            f"Point {index}: orig=({original_lon:.6f}, {original_lat:.6f}) -> "
            f"decoded=({decoded_lon:.6f}, {decoded_lat:.6f}){offset_suffix}"
        )

    if plot_graph:
        _visualize_bitpacked_colormap(centroids)

    return h3_dict


def _visualize_bitpacked_colormap(centroids: pd.DataFrame) -> None:
    figure_one = plt.figure(figsize=(20, 12))
    axis_one = figure_one.add_subplot(1, 1, 1)
    axis_one.scatter(
        centroids["avg_lon"],
        centroids["avg_lat"],
        c=centroids["color_plot"].tolist(),
        s=10,
        alpha=0.8,
    )
    axis_one.set_xlabel("Longitude ->", fontsize=12, fontweight="bold")
    axis_one.set_ylabel("Latitude ->", fontsize=12, fontweight="bold")
    axis_one.set_title(
        "Geographic Distribution (Bit-Packed RGB)\nR=lat_high, G=lat_low|lon_low, B=lon_high",
        fontsize=13,
        fontweight="bold",
    )
    axis_one.grid(True, alpha=0.3)

    figure_two = plt.figure(figsize=(20, 12))
    axis_two = figure_two.add_subplot(1, 1, 1)
    axis_two.scatter(
        centroids["X"],
        centroids["Y"],
        c=centroids["color_plot"].tolist(),
        s=10,
        alpha=0.8,
    )
    axis_two.set_xlabel("Normalized Longitude [0,1]", fontsize=11)
    axis_two.set_ylabel("Normalized Latitude [0,1]", fontsize=11)
    axis_two.set_title(
        "Normalized Space (Bit-Packed)\nColors encode precise coordinates",
        fontsize=13,
        fontweight="bold",
    )
    axis_two.grid(True, alpha=0.3)
    axis_two.set_xlim(-0.05, 1.05)
    axis_two.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig("float32_bitpacked_colormap.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to: float32_bitpacked_colormap.png")
    plt.show()


def build_h3_color_and_position_maps(
    augmented_trip_df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    h3_col: str = "h3",
    plot_graph: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[float, float, float]], dict[str, tuple[float, float]]]:
    h3_dict = generate_float32_bitpacked_colormap(
        augmented_trip_df,
        lon_col=lon_col,
        lat_col=lat_col,
        h3_col=h3_col,
        plot_graph=plot_graph,
    )
    h3_nodes = list(h3_dict.keys())
    h3_color_map = {
        h3_cell: tuple(float(component) for component in h3_dict[h3_cell]["color"])
        for h3_cell in h3_nodes
    }
    h3_position_map = {
        h3_cell: tuple(float(component) for component in h3_dict[h3_cell]["position"])
        for h3_cell in h3_nodes
    }
    return h3_dict, h3_color_map, h3_position_map


def save_h3_maps_to_json(
    h3_color_map: dict[str, Sequence[float]],
    h3_position_map: dict[str, Sequence[float]],
    color_map_path: str | Path,
    position_map_path: str | Path,
) -> tuple[str, str]:
    color_output = _ensure_parent_dir(color_map_path)
    position_output = _ensure_parent_dir(position_map_path)

    serializable_colors = {
        h3_cell: [float(component) for component in color]
        for h3_cell, color in h3_color_map.items()
    }
    serializable_positions = {
        h3_cell: [float(component) for component in position]
        for h3_cell, position in h3_position_map.items()
    }

    with open(color_output, "w", encoding="utf-8") as handle:
        json.dump(serializable_colors, handle, indent=2)
    with open(position_output, "w", encoding="utf-8") as handle:
        json.dump(serializable_positions, handle, indent=2)

    return color_output.as_posix(), position_output.as_posix()


def load_h3_maps_from_json(
    color_map_path: str | Path,
    position_map_path: str | Path,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float]]]:
    with open(color_map_path, "r", encoding="utf-8") as handle:
        color_map_raw = json.load(handle)
    with open(position_map_path, "r", encoding="utf-8") as handle:
        position_map_raw = json.load(handle)

    h3_color_map = {
        str(h3_cell): tuple(float(component) for component in color)
        for h3_cell, color in color_map_raw.items()
    }
    h3_position_map = {
        str(h3_cell): tuple(float(component) for component in position)
        for h3_cell, position in position_map_raw.items()
    }
    return h3_color_map, h3_position_map


def create_wave_map_with_missing(
    df: pd.DataFrame,
    h3_color_map: Optional[dict[str, Sequence[float]]] = None,
    h3_position_map: Optional[dict[str, Sequence[float]]] = None,
    missing_bins: Optional[Sequence[tuple[int, int]]] = None,
    bins: int = 128,
    output_file: Optional[str | Path] = None,
    longest_trip_duration_seconds: Optional[float] = None,
    time_col: str = "time",
    lon_col: str = "lon",
    lat_col: str = "lat",
    h3_resolution: int = 10,
    mask_after_trip_end: bool = True,
    save_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    height = width = n_bins = bins
    if h3_color_map is None:
        raise ValueError("h3_color_map must be provided")
    if h3_position_map is None:
        raise ValueError("h3_position_map must be provided")

    trip_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trip_df[time_col]):
        trip_df[time_col] = parse_trip_times(trip_df[time_col])

    trip_start = trip_df[time_col].min()
    trip_end = trip_df[time_col].max()
    trip_df["elapsed_seconds"] = (trip_df[time_col] - trip_start).dt.total_seconds()

    current_trip_duration = (trip_end - trip_start).total_seconds()
    if longest_trip_duration_seconds is None:
        longest_trip_duration_seconds = current_trip_duration

    image_array = np.zeros((3, height, width), dtype=np.float32)
    mask_array = np.zeros((1, height, width), dtype=np.uint8)
    bin_duration = longest_trip_duration_seconds / n_bins

    h3_cells = list(h3_color_map.keys())
    h3_positions_array = np.array([h3_position_map[cell] for cell in h3_cells], dtype=np.float32)

    def get_color_for_lonlat(lon_value: float, lat_value: float) -> np.ndarray:
        cell = h3.latlng_to_cell(lat_value, lon_value, h3_resolution)
        if cell in h3_color_map:
            return np.array(h3_color_map[cell], dtype=np.float32)
        distances = (h3_positions_array[:, 0] - lon_value) ** 2 + (h3_positions_array[:, 1] - lat_value) ** 2
        closest_index = int(np.argmin(distances))
        closest_cell = h3_cells[closest_index]
        return np.array(h3_color_map[closest_cell], dtype=np.float32)

    last_row = trip_df.loc[trip_df["elapsed_seconds"].idxmax()]
    last_color = get_color_for_lonlat(last_row[lon_col], last_row[lat_col])

    for bin_index in range(n_bins):
        bin_start = bin_index * bin_duration
        bin_end = (bin_index + 1) * bin_duration
        row_index = int((bin_index / n_bins) * height)

        if bin_start >= current_trip_duration:
            if mask_after_trip_end:
                mask_array[0, row_index, :] = 255
                image_array[:, row_index, :] = 1.0
            else:
                image_array[:, row_index, :] = last_color[:, np.newaxis]
            continue

        bin_data = trip_df[
            (trip_df["elapsed_seconds"] >= bin_start) & (trip_df["elapsed_seconds"] < bin_end)
        ]
        if bin_data.empty:
            mask_array[0, row_index, :] = 255
            image_array[:, row_index, :] = 1.0
            continue

        internal_times = np.linspace(bin_start, bin_end, width)
        elapsed_values = bin_data["elapsed_seconds"].values
        for column_index, internal_time in enumerate(internal_times):
            closest_index = int(np.abs(elapsed_values - internal_time).argmin())
            closest_row = bin_data.iloc[closest_index]
            color = get_color_for_lonlat(closest_row[lon_col], closest_row[lat_col])
            image_array[:, row_index, column_index] = color

    if missing_bins:
        for start_bin, end_bin in missing_bins:
            safe_start = max(0, int(start_bin))
            safe_end = min(n_bins, int(end_bin))
            if safe_end > safe_start:
                start_row = int((safe_start / n_bins) * height)
                end_row = int((safe_end / n_bins) * height)
                mask_array[0, start_row:end_row, :] = 255
                image_array[:, start_row:end_row, :] = 1.0

    if output_file is not None:
        output_base = _as_path(output_file)
        if output_base.suffix:
            output_base = output_base.with_suffix("")
        np.save(output_base.with_suffix(".npy"), image_array)
        if save_mask:
            np.save(output_base.parent / f"{output_base.name}_mask.npy", mask_array)

    return image_array, mask_array


def save_wave_maps_for_all_trips(
    trip_data: str | Path | pd.DataFrame,
    image_dir: str | Path,
    h3_color_map: dict[str, Sequence[float]],
    h3_position_map: dict[str, Sequence[float]],
    longest_trip_duration_seconds: Optional[float] = None,
    h3_resolution: int = 10,
    n_bins: int = 128,
    save_mask: bool = False,
    verbose: bool = True,
    rename_map: Optional[Mapping[str, str]] = None,
) -> tuple[list[str], float]:
    trip_df = prepare_trip_df(
        trip_data,
        required_columns=TRIP_REQUIRED_COLUMNS,
        h3_resolution=h3_resolution,
        rename_map=rename_map,
    )
    image_root = _as_path(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)

    effective_longest_duration = (
        longest_trip_duration_seconds
        if longest_trip_duration_seconds is not None
        else compute_longest_trip_duration_seconds(trip_df)
    )
    if verbose:
        print(
            f"Longest trip duration: {effective_longest_duration:.2f} seconds "
            f"({effective_longest_duration / 3600:.2f} hours)"
        )

    saved_paths: list[str] = []
    for trip_id in trip_df["trip_id"].dropna().unique():
        trip_slice = (
            trip_df[trip_df["trip_id"] == trip_id]
            .sort_values("time")
            .reset_index(drop=True)
        )
        if verbose:
            print(
                f"Trip {trip_id}: {len(trip_slice)} time steps, {trip_slice['h3'].nunique()} unique H3 cells"
            )
        output_base = image_root / f"wave_map_trip_{trip_id}"
        create_wave_map_with_missing(
            trip_slice,
            h3_color_map=h3_color_map,
            h3_position_map=h3_position_map,
            missing_bins=None,
            bins=n_bins,
            output_file=output_base,
            longest_trip_duration_seconds=effective_longest_duration,
            time_col="time",
            lon_col="lon",
            lat_col="lat",
            h3_resolution=h3_resolution,
            save_mask=save_mask,
        )
        saved_paths.append(output_base.with_suffix(".npy").as_posix())

    return saved_paths, float(effective_longest_duration)


def load_wave_map_image(image_path: str | Path) -> np.ndarray:
    image_array = np.load(image_path).astype(np.float32)
    if image_array.ndim == 3 and image_array.shape[0] != 3:
        image_array = image_array.transpose(2, 0, 1)
    return np.clip(image_array, 0, 1).astype(np.float32)


def list_wave_map_images(image_dir: str | Path) -> list[str]:
    image_root = _as_path(image_dir)
    return sorted(
        _normalize_existing_path(path)
        for path in image_root.glob("wave_map_trip_*.npy")
        if not path.name.endswith("_mask.npy")
    )


def _canonical_holdout_image_key(image_path: str | Path) -> str:
    resolved_path = _as_path(image_path).resolve()
    parts = resolved_path.parts

    for index in range(len(parts) - 2, -1, -1):
        if parts[index].startswith("images"):
            return Path(*parts[index:]).as_posix()

    return resolved_path.as_posix()


def normalize_holdout_percentage(value: float) -> float:
    normalized_value = value / 100.0 if value > 1.0 else value
    if not 0 < normalized_value < 1:
        raise ValueError("holdout_percentage must be between 0 and 1, or between 0 and 100.")
    return float(normalized_value)


def split_holdout_paths(
    image_paths_or_dir: str | Path | Sequence[str],
    holdout_percentage: float = 0.30,
    salt: str = "archimedes_holdout",
) -> tuple[list[str], list[str]]:
    effective_percentage = normalize_holdout_percentage(holdout_percentage)
    if isinstance(image_paths_or_dir, (str, Path)):
        image_paths = list_wave_map_images(image_paths_or_dir)
    else:
        image_paths = sorted(_normalize_existing_path(path) for path in image_paths_or_dir)

    if len(image_paths) < 2:
        raise ValueError("At least two images are required to create a holdout split.")

    holdout_size = max(1, int(np.ceil(len(image_paths) * effective_percentage)))
    holdout_size = min(holdout_size, len(image_paths) - 1)
    canonical_keys = {path: _canonical_holdout_image_key(path) for path in image_paths}

    ranked_paths = sorted(
        image_paths,
        key=lambda path: (
            hashlib.sha256(f"{salt}:{canonical_keys[path]}".encode("utf-8")).hexdigest(),
            canonical_keys[path],
            path,
        ),
    )
    holdout_paths = sorted(ranked_paths[:holdout_size])
    holdout_lookup = set(holdout_paths)
    train_paths = sorted(path for path in image_paths if path not in holdout_lookup)
    return train_paths, holdout_paths


def write_holdout_images_file(
    holdout_paths: Sequence[str | Path],
    output_path: str | Path,
    header: str = "# Holdout images (NOT used in training)",
    relative_to: Optional[str | Path] = None,
) -> str:
    output_file = _ensure_parent_dir(output_path)
    relative_root = _as_path(relative_to).resolve() if relative_to is not None else None

    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(f"{header}\n")
        for holdout_path in holdout_paths:
            path_obj = _as_path(holdout_path)
            if relative_root is not None and path_obj.is_absolute():
                display_path = Path(os.path.relpath(path_obj, relative_root)).as_posix()
            else:
                display_path = path_obj.as_posix()
            handle.write(f"{display_path}\n")
    return output_file.as_posix()


def read_holdout_images_file(holdout_file: str | Path) -> list[str]:
    with open(holdout_file, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and line.strip().endswith(".npy")
        ]


def trip_id_from_image_path(image_path: str | Path) -> str:
    filename = _as_path(image_path).name
    prefix = "wave_map_trip_"
    if not filename.startswith(prefix) or not filename.endswith(".npy"):
        raise ValueError(f"Unsupported wave-map filename: {filename}")
    return filename[len(prefix) : -4]


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_checkpoint_model_state(
    checkpoint_path: str | Path,
    map_location: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")


def _build_masked_inpainting_input(
    image_array: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masked_image = image_array.copy()
    mask_three_channel = np.repeat(mask, 3, axis=0)
    masked_image[mask_three_channel == 0] = 1.0
    input_array = np.concatenate([masked_image, mask], axis=0).astype(np.float32)
    return masked_image, input_array


def _run_inpainting_model(
    model: nn.Module,
    input_array: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    input_tensor = torch.from_numpy(input_array).float().unsqueeze(0).to(device)
    try:
        model.eval()
        use_amp = device.type == "cuda"
        if use_amp:
            with torch.amp.autocast("cuda"):
                output = model(input_tensor)
        else:
            output = model(input_tensor)
        return output.squeeze(0).cpu().numpy().astype(np.float32)
    finally:
        del input_tensor


def _compute_inpainting_losses(
    output: torch.Tensor,
    batch_target: torch.Tensor,
    batch_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mse_loss = (output - batch_target) ** 2
    l1_loss = torch.abs(output - batch_target)
    missing_mask = (batch_mask == 0).expand_as(mse_loss)
    full_loss = mse_loss.mean() + l1_loss.mean()
    if bool(missing_mask.any()):
        masked_loss = mse_loss[missing_mask].mean() + l1_loss[missing_mask].mean()
    else:
        masked_loss = full_loss
    total_loss = 0.2 * full_loss + 0.8 * masked_loss
    return total_loss, masked_loss


class H3ColorQuantizer:
    def __init__(self, h3_color_map: dict[str, Sequence[float]]):
        self.h3_cells = list(h3_color_map.keys())
        self.colors = np.array([h3_color_map[cell] for cell in self.h3_cells], dtype=np.float32)
        self.n_colors = len(self.h3_cells)
        self.tree = cKDTree(self.colors)
        print(f"H3 Color Quantizer initialized with {self.n_colors} valid colors (float32)")

    def quantize(self, rgb_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        original_shape = rgb_array.shape
        flat_rgb = rgb_array.reshape(-1, 3).astype(np.float32)
        _, indices = self.tree.query(flat_rgb)
        quantized_flat = self.colors[indices]
        return quantized_flat.reshape(original_shape), indices.reshape(original_shape[:-1])

    def quantize_image(self, img_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image_hwc = img_chw.transpose(1, 2, 0).astype(np.float32)
        quantized_hwc, indices = self.quantize(image_hwc)
        return quantized_hwc.transpose(2, 0, 1), indices

    def get_h3_cells_from_indices(self, indices: np.ndarray) -> np.ndarray:
        return np.array([self.h3_cells[index] for index in indices.flat]).reshape(indices.shape)


class H3InpaintingModel(nn.Module):
    def __init__(self, in_channels: int = 4, base_ch: int = 64):
        super().__init__()
        self.enc1 = self._double_conv(in_channels, base_ch)
        self.enc2 = self._double_conv(base_ch, base_ch * 2)
        self.enc3 = self._double_conv(base_ch * 2, base_ch * 4)
        self.enc4 = self._double_conv(base_ch * 4, base_ch * 8)
        self.bottleneck = self._double_conv(base_ch * 8, base_ch * 16)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = self._double_conv(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = self._double_conv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = self._double_conv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = self._double_conv(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.1)

    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder_one = self.enc1(inputs)
        encoder_two = self.enc2(self.pool(encoder_one))
        encoder_three = self.enc3(self.pool(encoder_two))
        encoder_four = self.enc4(self.pool(encoder_three))
        bottleneck = self.dropout(self.bottleneck(self.pool(encoder_four)))
        decoder_four = self.dec4(torch.cat([self.up4(bottleneck), encoder_four], dim=1))
        decoder_three = self.dec3(torch.cat([self.up3(decoder_four), encoder_three], dim=1))
        decoder_two = self.dec2(torch.cat([self.up2(decoder_three), encoder_two], dim=1))
        decoder_one = self.dec1(torch.cat([self.up1(decoder_two), encoder_one], dim=1))
        return torch.sigmoid(self.out_conv(decoder_one))


class H3InpaintDatasetAugmented(Dataset):
    def __init__(self, image_dir: str | Path, exclude_paths: Optional[Sequence[str | Path]] = None):
        all_paths = [
            path.as_posix()
            for path in _as_path(image_dir).glob("wave_map_trip_*.npy")
            if not path.name.endswith("_mask.npy")
        ]
        exclude_lookup = (
            {_normalize_existing_path(path) for path in exclude_paths}
            if exclude_paths is not None
            else set()
        )
        self.image_paths = [
            path for path in all_paths if _normalize_existing_path(path) not in exclude_lookup
        ]
        self.mask_configs = [
            (0.05, 0.08, 200),
            (0.08, 0.12, 100),
            (0.12, 0.16, 50),
            (0.16, 0.22, 25),
            (0.22, 0.30, 10),
        ]
        self.augmentations = []
        for config_index, (_, _, num_positions) in enumerate(self.mask_configs):
            for position_index in range(num_positions):
                self.augmentations.append((config_index, position_index, num_positions))
        self.total_augs = len(self.augmentations)
        print(
            f"Found {len(self.image_paths)} images x {self.total_augs} augmentations = {len(self)} samples"
        )
        print(
            f"Mask configurations: {len(self.mask_configs)} sizes, positions per size: {[config[2] for config in self.mask_configs]}"
        )

    def __len__(self) -> int:
        return len(self.image_paths) * self.total_augs

    def _create_mask(
        self,
        height: int,
        width: int,
        config_index: int,
        position_index: int,
        num_positions: int,
    ) -> np.ndarray:
        mask = np.ones((1, height, width), dtype=np.float32)
        min_pct, max_pct, _ = self.mask_configs[config_index]
        mask_pct = np.random.uniform(min_pct, max_pct)
        mask_height = int(height * mask_pct)
        margin = int(height * 0.05)
        valid_start_min = margin
        valid_start_max = height - margin - mask_height
        if valid_start_max <= valid_start_min:
            valid_start_max = max(valid_start_min + 1, height - mask_height)
        valid_range = valid_start_max - valid_start_min
        zone_size = max(1, valid_range // num_positions)
        zone_start = valid_start_min + position_index * zone_size
        zone_end = min(zone_start + zone_size, valid_start_max)
        if zone_end > zone_start:
            start = int(np.random.randint(zone_start, zone_end + 1))
        else:
            start = zone_start
        end = min(start + mask_height, height - margin)
        mask[0, start:end, :] = 0
        return mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        image_index = index // self.total_augs
        augmentation_index = index % self.total_augs
        config_index, position_index, num_positions = self.augmentations[augmentation_index]
        image_array = load_wave_map_image(self.image_paths[image_index])
        _, height, width = image_array.shape
        mask = self._create_mask(height, width, config_index, position_index, num_positions)
        _, input_tensor = _build_masked_inpainting_input(image_array, mask)
        return (
            torch.from_numpy(input_tensor).float(),
            torch.from_numpy(image_array).float(),
            torch.from_numpy(mask).float(),
            self.image_paths[image_index],
        )


def train_h3_inpainting(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 2e-4,
    save_path: str | Path = "h3_inpainting.pth",
    num_workers: int = 4,
) -> tuple[nn.Module, dict[str, list[float]]]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Generate images before training.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    history = {"loss": [], "masked_loss": []}
    best_loss = float("inf")
    save_file = _ensure_parent_dir(save_path)

    print(f"\nTraining H3 RGB Inpainting (float32) for {epochs} epochs...")
    print(f"Dataset size: {len(dataset)}, Batch: {batch_size}, LR: {lr}")
    print(f"Device: {device}, AMP enabled: {use_amp}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_masked_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch_input, batch_target, batch_mask, _ in progress_bar:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    output = model(batch_input)
                    loss, masked_loss = _compute_inpainting_losses(output, batch_target, batch_mask)
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(batch_input)
                loss, masked_loss = _compute_inpainting_losses(output, batch_target, batch_mask)
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss += float(loss.item())
            epoch_masked_loss += float(masked_loss.item())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        avg_masked_loss = epoch_masked_loss / len(dataloader)
        history["loss"].append(avg_loss)
        history["masked_loss"].append(avg_masked_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state": model.state_dict()}, save_file)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Masked={avg_masked_loss:.4f}, "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )
        clear_memory()

    model.load_state_dict(load_checkpoint_model_state(save_file, map_location=device))
    print(f"\nBest model loaded (loss={best_loss:.4f})")
    return model, history


def plot_training_history(
    history: dict[str, Sequence[float]],
    output_path: Optional[str | Path] = None,
    title: str = "Training Loss",
) -> plt.Figure:
    figure, axis = plt.subplots(1, 1, figsize=(10, 4))
    axis.plot(history.get("loss", []), "b-", label="Total Loss")
    axis.plot(history.get("masked_loss", []), "r-", label="Masked Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss (L1 + MSE)")
    axis.set_title(title)
    axis.legend()
    axis.grid(True)
    plt.tight_layout()
    if output_path is not None:
        output_file = _ensure_parent_dir(output_path)
        figure.savefig(output_file, dpi=150)
    return figure


@torch.no_grad()
def inpaint_h3(
    model: nn.Module,
    quantizer: H3ColorQuantizer,
    image_path: str | Path,
    device: torch.device,
    missing_rows: tuple[int, int] = (60, 70),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clear_memory()
    image_array = load_wave_map_image(image_path)
    _, height, _ = image_array.shape
    start_row, end_row = missing_rows
    start_row = max(0, min(start_row, height - 1))
    end_row = max(start_row + 1, min(end_row, height))

    mask = np.ones((1, height, image_array.shape[2]), dtype=np.float32)
    mask[0, start_row:end_row, :] = 0
    masked_image, input_array = _build_masked_inpainting_input(image_array, mask)
    rgb_output = _run_inpainting_model(model, input_array, device)

    quantized_rgb, pred_classes = quantizer.quantize_image(rgb_output)
    final_image = image_array.copy()
    final_image[:, start_row:end_row, :] = quantized_rgb[:, start_row:end_row, :]
    h3_cells = quantizer.get_h3_cells_from_indices(pred_classes)
    clear_memory()
    return final_image, mask[0], masked_image, pred_classes, h3_cells, rgb_output


def compute_h3_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    pred_classes: np.ndarray,
    true_classes: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    metrics["rmse_full"] = float(np.sqrt(np.mean((original - reconstructed) ** 2)))
    mask_three_channel = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
    original_masked = original[mask_three_channel == 0]
    reconstructed_masked = reconstructed[mask_three_channel == 0]
    metrics["rmse_masked"] = float(np.sqrt(np.mean((original_masked - reconstructed_masked) ** 2)))
    masked_region = mask == 0
    if masked_region.sum() > 0:
        correct = int((pred_classes[masked_region] == true_classes[masked_region]).sum())
        total = int(masked_region.sum())
        metrics["accuracy"] = float(correct / total)
        metrics["error_rate"] = 1.0 - metrics["accuracy"]
    return metrics


def evaluate_holdout_images(
    model: nn.Module,
    quantizer: H3ColorQuantizer,
    holdout_paths: Sequence[str | Path],
    device: torch.device,
    missing_fraction_ranges: Sequence[tuple[float, float]] = ((0.30, 0.40), (0.40, 0.55), (0.50, 0.70)),
) -> list[dict[str, Any]]:
    all_metrics: list[dict[str, Any]] = []
    for image_path in holdout_paths:
        original = load_wave_map_image(image_path)
        _, true_classes = quantizer.quantize_image(original)
        height = original.shape[1]
        test_masks = [
            (int(height * start_fraction), int(height * end_fraction))
            for start_fraction, end_fraction in missing_fraction_ranges
        ]
        for missing_rows in test_masks:
            reconstructed, mask, masked_input, pred_classes, h3_cells, rgb_output = inpaint_h3(
                model,
                quantizer,
                image_path,
                device,
                missing_rows=missing_rows,
            )
            metrics = compute_h3_metrics(
                original,
                reconstructed,
                mask,
                pred_classes,
                true_classes,
            )
            metrics.update(
                {
                    "image": _as_path(image_path).name,
                    "missing_rows": missing_rows,
                    "is_holdout": True,
                    "masked_input": masked_input,
                    "h3_cells": h3_cells,
                    "rgb_output": rgb_output,
                }
            )
            all_metrics.append(metrics)
        clear_memory()
    return all_metrics


def fill_small_gaps_interpolation(
    trip_df: pd.DataFrame,
    trip_id: str,
    h3_resolution: int = 10,
    small_gap_threshold_seconds: float = 120,
    verbose: bool = True,
    rename_map: Optional[Mapping[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, int], list[list[int]]]:
    trip_data = prepare_trip_df(
        trip_df,
        required_columns=TRIP_REQUIRED_COLUMNS,
        h3_resolution=h3_resolution,
        rename_map=rename_map,
    )
    trip_data = _select_trip_rows(trip_data, trip_id)

    if "fill_source" not in trip_data.columns:
        trip_data["fill_source"] = "original"
    trip_data.loc[~(trip_data["lon"].isna() | trip_data["lat"].isna()), "fill_source"] = "original"

    missing_mask = trip_data["lon"].isna() | trip_data["lat"].isna()
    if missing_mask.sum() == 0:
        if verbose:
            print("No missing data to fill.")
        return trip_data, {"small_gaps_filled": 0, "small_segments_filled": 0}, []

    missing_indices = trip_data[missing_mask].index.tolist()
    segments = []
    current_segment = [missing_indices[0]]
    for index in range(1, len(missing_indices)):
        if missing_indices[index] == missing_indices[index - 1] + 1:
            current_segment.append(missing_indices[index])
        else:
            segments.append(current_segment)
            current_segment = [missing_indices[index]]
    segments.append(current_segment)

    if verbose:
        print(f"Found {len(segments)} missing segments")

    stats = {"small_gaps_filled": 0, "small_segments_filled": 0}
    large_gap_segments: list[list[int]] = []

    for segment_index, segment in enumerate(segments):
        start_index = segment[0]
        end_index = segment[-1]
        previous_index, next_index = _find_known_trip_neighbors(trip_data, start_index, end_index)

        previous_time = trip_data.loc[previous_index, "time"] if previous_index >= 0 else None
        previous_lon = trip_data.loc[previous_index, "lon"] if previous_index >= 0 else None
        previous_lat = trip_data.loc[previous_index, "lat"] if previous_index >= 0 else None
        next_time = trip_data.loc[next_index, "time"] if next_index < len(trip_data) else None
        next_lon = trip_data.loc[next_index, "lon"] if next_index < len(trip_data) else None
        next_lat = trip_data.loc[next_index, "lat"] if next_index < len(trip_data) else None

        if previous_time is not None and next_time is not None:
            total_gap_duration = (next_time - previous_time).total_seconds()
        elif previous_time is not None:
            total_gap_duration = (trip_data.loc[end_index, "time"] - previous_time).total_seconds()
        elif next_time is not None:
            total_gap_duration = (next_time - trip_data.loc[start_index, "time"]).total_seconds()
        else:
            total_gap_duration = float("inf")

        if total_gap_duration < small_gap_threshold_seconds and previous_lon is not None and next_lon is not None:
            for row_index in segment:
                target_time = trip_data.loc[row_index, "time"]
                interpolated_lon, interpolated_lat = _interpolate_trip_coords(
                    previous_time,
                    previous_lon,
                    previous_lat,
                    next_time,
                    next_lon,
                    next_lat,
                    target_time,
                )
                try:
                    interpolated_h3 = h3.latlng_to_cell(interpolated_lat, interpolated_lon, h3_resolution)
                except Exception:
                    interpolated_h3 = None
                trip_data.loc[row_index, "lon"] = interpolated_lon
                trip_data.loc[row_index, "lat"] = interpolated_lat
                trip_data.loc[row_index, "h3"] = interpolated_h3
                trip_data.loc[row_index, "fill_source"] = "small_gap_interp"
                stats["small_gaps_filled"] += 1
            stats["small_segments_filled"] += 1
            if verbose:
                print(
                    f"Segment {segment_index + 1}: {len(segment)} points, {total_gap_duration:.0f}s -> FILLED"
                )
        else:
            large_gap_segments.append(segment)
            if verbose:
                print(
                    f"Segment {segment_index + 1}: {len(segment)} points, {total_gap_duration:.0f}s -> LARGE GAP"
                )

    if verbose:
        print("\n=== Stage 1 Summary ===")
        print(
            f"Small gaps filled: {stats['small_gaps_filled']} points in {stats['small_segments_filled']} segments"
        )
        print(
            f"Large gaps remaining: {len(large_gap_segments)} segments, {sum(len(segment) for segment in large_gap_segments)} points"
        )

    return trip_data, stats, large_gap_segments


def fill_large_gaps_from_inpainted_image(
    trip_df: pd.DataFrame,
    large_gap_segments: Sequence[Sequence[int]],
    inpainted_img: np.ndarray,
    h3_color_map: dict[str, Sequence[float]],
    h3_position_map: dict[str, Sequence[float]],
    trip_start: pd.Timestamp,
    longest_trip_duration_seconds: float,
    h3_resolution: int = 10,
    sample_interval_seconds: float = 120,
    n_bins: int = 128,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    trip_data = trip_df.copy()
    if len(large_gap_segments) == 0:
        if verbose:
            print("No large gaps to fill.")
        return trip_data, {"large_gaps_filled": 0, "large_segments_filled": 0}

    all_colors = np.array(list(h3_color_map.values()), dtype=np.float32)
    all_h3_ids = list(h3_color_map.keys())
    height, width = inpainted_img.shape[1], inpainted_img.shape[2]
    bin_duration = longest_trip_duration_seconds / n_bins
    stats = {"large_gaps_filled": 0, "large_segments_filled": 0}

    def get_median_coords_from_image_row(elapsed_seconds: float) -> tuple[Optional[float], Optional[float], Optional[str]]:
        bin_index = int(elapsed_seconds / bin_duration)
        bin_index = max(0, min(bin_index, n_bins - 1))
        row_index = int((bin_index / n_bins) * height)
        row_index = max(0, min(row_index, height - 1))
        row_pixels = inpainted_img[:, row_index, :].T
        coords_in_row = []
        for column_index in range(width):
            pixel_color = row_pixels[column_index].astype(np.float32)
            if np.allclose(pixel_color, [1.0, 1.0, 1.0], atol=0.01):
                continue
            distances = np.linalg.norm(all_colors - pixel_color, axis=1)
            closest_index = int(np.argmin(distances))
            min_distance = float(distances[closest_index])
            if min_distance < 0.1:
                h3_id = all_h3_ids[closest_index]
                position = h3_position_map.get(h3_id)
                if position is not None:
                    coords_in_row.append((position[0], position[1], h3_id))
        if len(coords_in_row) == 0:
            return None, None, None
        lon_values = [value[0] for value in coords_in_row]
        lat_values = [value[1] for value in coords_in_row]
        median_lon = float(np.median(lon_values))
        median_lat = float(np.median(lat_values))
        try:
            median_h3 = h3.latlng_to_cell(median_lat, median_lon, h3_resolution)
        except Exception:
            median_h3 = coords_in_row[len(coords_in_row) // 2][2]
        return median_lon, median_lat, median_h3

    for segment_index, segment in enumerate(large_gap_segments):
        start_index = segment[0]
        end_index = segment[-1]
        previous_index, next_index = _find_known_trip_neighbors(trip_data, start_index, end_index)
        if previous_index < 0 or next_index >= len(trip_data):
            if verbose:
                print(f"Large segment {segment_index + 1}: SKIPPED (missing boundary points)")
            continue

        previous_time = trip_data.loc[previous_index, "time"]
        previous_lon = trip_data.loc[previous_index, "lon"]
        previous_lat = trip_data.loc[previous_index, "lat"]
        previous_h3 = trip_data.loc[previous_index, "h3"]
        next_time = trip_data.loc[next_index, "time"]
        next_lon = trip_data.loc[next_index, "lon"]
        next_lat = trip_data.loc[next_index, "lat"]
        next_h3 = trip_data.loc[next_index, "h3"]

        anchor_points = [(previous_time, previous_lon, previous_lat, previous_h3)]
        sample_time = previous_time + pd.Timedelta(seconds=sample_interval_seconds)
        while sample_time < next_time:
            elapsed_seconds = (sample_time - trip_start).total_seconds()
            sample_lon, sample_lat, sample_h3 = get_median_coords_from_image_row(elapsed_seconds)
            if sample_lon is not None:
                anchor_points.append((sample_time, sample_lon, sample_lat, sample_h3))
            sample_time += pd.Timedelta(seconds=sample_interval_seconds)
        anchor_points.append((next_time, next_lon, next_lat, next_h3))

        if verbose:
            print(
                f"Large segment {segment_index + 1}: {len(segment)} points, {len(anchor_points)} anchors from inpainted image"
            )

        filled_in_segment = 0
        for row_index in segment:
            target_time = trip_data.loc[row_index, "time"]
            left_anchor = None
            right_anchor = None
            for anchor_index in range(len(anchor_points) - 1):
                if anchor_points[anchor_index][0] <= target_time <= anchor_points[anchor_index + 1][0]:
                    left_anchor = anchor_points[anchor_index]
                    right_anchor = anchor_points[anchor_index + 1]
                    break
            if left_anchor is None or right_anchor is None:
                continue
            interpolated_lon, interpolated_lat = _interpolate_trip_coords(
                left_anchor[0],
                left_anchor[1],
                left_anchor[2],
                right_anchor[0],
                right_anchor[1],
                right_anchor[2],
                target_time,
            )
            try:
                interpolated_h3 = h3.latlng_to_cell(interpolated_lat, interpolated_lon, h3_resolution)
            except Exception:
                interpolated_h3 = None
            trip_data.loc[row_index, "lon"] = interpolated_lon
            trip_data.loc[row_index, "lat"] = interpolated_lat
            trip_data.loc[row_index, "h3"] = interpolated_h3
            trip_data.loc[row_index, "fill_source"] = "large_gap_inpainted"
            stats["large_gaps_filled"] += 1
            filled_in_segment += 1

        if filled_in_segment > 0:
            stats["large_segments_filled"] += 1
        if verbose:
            print(f" -> Filled {filled_in_segment}/{len(segment)} points")

    if verbose:
        print("\n=== Stage 3 Summary ===")
        print(
            f"Large gaps filled: {stats['large_gaps_filled']} points in {stats['large_segments_filled']} segments"
        )
    return trip_data, stats


@torch.no_grad()
def run_inpainting_inference(
    trip_df: pd.DataFrame,
    trip_id: str,
    h3_color_map: dict[str, Sequence[float]],
    h3_position_map: dict[str, Sequence[float]],
    model_path: str | Path = "h3_rgb_unet_v2.pth",
    longest_trip_duration_seconds: Optional[float] = None,
    n_bins: int = 128,
    device: Optional[torch.device] = None,
    base_ch: int = 48,
    h3_resolution: int = 10,
    rename_map: Optional[Mapping[str, str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trip_data = prepare_trip_df(
        trip_df,
        required_columns=TRIP_REQUIRED_COLUMNS,
        h3_resolution=h3_resolution,
        rename_map=rename_map,
    )
    trip_data = _select_trip_rows(trip_data, trip_id)

    trip_start = trip_data["time"].min()
    trip_end = trip_data["time"].max()
    if longest_trip_duration_seconds is None:
        longest_trip_duration_seconds = (trip_end - trip_start).total_seconds()

    missing_mask = trip_data["lon"].isna() | trip_data["lat"].isna() | trip_data["h3"].isna()
    known_data = trip_data[~missing_mask].copy()

    trip_data["elapsed_seconds"] = (trip_data["time"] - trip_start).dt.total_seconds()
    bin_duration = longest_trip_duration_seconds / n_bins
    missing_indices = trip_data[missing_mask].index.tolist()
    missing_bins = set()
    for row_index in missing_indices:
        elapsed_seconds = trip_data.loc[row_index, "elapsed_seconds"]
        bin_index = int(elapsed_seconds / bin_duration)
        bin_index = max(0, min(bin_index, n_bins - 1))
        missing_bins.add(bin_index)

    missing_bins_sorted = sorted(missing_bins)
    missing_bin_ranges = []
    if missing_bins_sorted:
        start_bin = missing_bins_sorted[0]
        end_bin = start_bin + 1
        for bin_index in missing_bins_sorted[1:]:
            if bin_index == end_bin:
                end_bin = bin_index + 1
            else:
                missing_bin_ranges.append((start_bin, end_bin))
                start_bin = bin_index
                end_bin = bin_index + 1
        missing_bin_ranges.append((start_bin, end_bin))

    image_array, mask_array = create_wave_map_with_missing(
        known_data,
        h3_color_map=h3_color_map,
        h3_position_map=h3_position_map,
        missing_bins=missing_bin_ranges,
        bins=n_bins,
        longest_trip_duration_seconds=longest_trip_duration_seconds,
        time_col="time",
        mask_after_trip_end=True,
        save_mask=False,
    )

    quantizer = H3ColorQuantizer(h3_color_map)
    model = H3InpaintingModel(in_channels=4, base_ch=base_ch).to(device)
    model.load_state_dict(load_checkpoint_model_state(model_path, map_location=device))

    image_array = image_array.astype(np.float32)
    mask = (mask_array[0] == 0).astype(np.float32)[np.newaxis, :, :]
    _, input_array = _build_masked_inpainting_input(image_array, mask)
    rgb_output = _run_inpainting_model(model, input_array, device)

    quantized_rgb, _ = quantizer.quantize_image(rgb_output)
    inpainted_image = image_array.copy()
    for channel_index in range(3):
        inpainted_image[channel_index][mask[0] == 0] = quantized_rgb[channel_index][mask[0] == 0]

    return inpainted_image, image_array, mask_array, trip_start


def run_gap_filling(
    trip_df: pd.DataFrame,
    trip_id: str,
    h3_color_map: dict[str, Sequence[float]],
    h3_position_map: dict[str, Sequence[float]],
    model_path: str | Path = "h3_rgb_unet_v2.pth",
    longest_trip_duration_seconds: Optional[float] = None,
    h3_resolution: int = 10,
    small_gap_threshold_seconds: float = 120,
    large_gap_sample_interval_seconds: float = 120,
    n_bins: int = 128,
    device: Optional[torch.device] = None,
    output_csv: Optional[str | Path] = None,
    verbose: bool = True,
    plot: bool = False,
    base_ch: int = 48,
    rename_map: Optional[Mapping[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if verbose:
        print("\n" + "=" * 70)
        print("TWO-STAGE GAP FILLING")
        print("=" * 70)

    trip_data = prepare_trip_df(
        trip_df,
        required_columns=TRIP_REQUIRED_COLUMNS,
        h3_resolution=h3_resolution,
        rename_map=rename_map,
    )
    trip_data = _select_trip_rows(trip_data, trip_id)

    trip_start = trip_data["time"].min()
    trip_end = trip_data["time"].max()
    if longest_trip_duration_seconds is None:
        longest_trip_duration_seconds = (trip_end - trip_start).total_seconds()

    initial_missing = int((trip_data["lon"].isna() | trip_data["lat"].isna()).sum())
    if verbose:
        print(f"Trip {trip_id}: {len(trip_data)} total rows, {initial_missing} initially missing")

    stage_one_df, stage_one_stats, large_gap_segments = fill_small_gaps_interpolation(
        trip_df=trip_data,
        trip_id=trip_id,
        h3_resolution=h3_resolution,
        small_gap_threshold_seconds=small_gap_threshold_seconds,
        verbose=verbose,
    )

    remaining_after_stage_one = int((stage_one_df["lon"].isna() | stage_one_df["lat"].isna()).sum())
    if verbose:
        print(f"After Stage 1: {remaining_after_stage_one} points still missing")

    inpainted_image = None
    original_image = None
    mask_array = None
    if remaining_after_stage_one > 0:
        inpainted_image, original_image, mask_array, _ = run_inpainting_inference(
            trip_df=stage_one_df,
            trip_id=trip_id,
            h3_color_map=h3_color_map,
            h3_position_map=h3_position_map,
            model_path=model_path,
            longest_trip_duration_seconds=longest_trip_duration_seconds,
            h3_resolution=h3_resolution,
            n_bins=n_bins,
            device=device,
            base_ch=base_ch,
        )
    elif verbose:
        print("No remaining gaps after Stage 1. Skipping inference.")

    if inpainted_image is not None and len(large_gap_segments) > 0:
        final_df, stage_three_stats = fill_large_gaps_from_inpainted_image(
            trip_df=stage_one_df,
            large_gap_segments=large_gap_segments,
            inpainted_img=inpainted_image,
            h3_color_map=h3_color_map,
            h3_position_map=h3_position_map,
            trip_start=trip_start,
            longest_trip_duration_seconds=longest_trip_duration_seconds,
            h3_resolution=h3_resolution,
            sample_interval_seconds=large_gap_sample_interval_seconds,
            n_bins=n_bins,
            verbose=verbose,
        )
    else:
        final_df = stage_one_df
        stage_three_stats = {"large_gaps_filled": 0, "large_segments_filled": 0}
        if verbose and len(large_gap_segments) == 0:
            print("No large gaps to fill from inpainted image.")

    final_missing = int((final_df["lon"].isna() | final_df["lat"].isna()).sum())
    total_filled = initial_missing - final_missing
    metrics = {
        "trip_id": trip_id,
        "total_rows": len(final_df),
        "initial_missing": initial_missing,
        "final_missing": final_missing,
        "total_filled": total_filled,
        "small_gaps_filled": stage_one_stats["small_gaps_filled"],
        "large_gaps_filled": stage_three_stats["large_gaps_filled"],
        "fill_rate": total_filled / initial_missing if initial_missing > 0 else 1.0,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("GAP FILLING SUMMARY")
        print("=" * 70)
        print(f"Total rows: {metrics['total_rows']}")
        print(f"Initially missing: {metrics['initial_missing']}")
        print(f"Small gaps filled (Stage 1): {metrics['small_gaps_filled']}")
        print(f"Large gaps filled (Stage 3): {metrics['large_gaps_filled']}")
        print(f"Total filled: {metrics['total_filled']}")
        print(f"Still missing: {metrics['final_missing']}")
        print(f"Fill rate: {metrics['fill_rate'] * 100:.1f}%")
        print("\nFill source distribution:")
        print(final_df["fill_source"].value_counts())

    if output_csv is not None:
        output_file = _ensure_parent_dir(output_csv)
        final_df.to_csv(output_file, index=False)

    if plot and inpainted_image is not None and original_image is not None and mask_array is not None:
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        original_display = np.clip(original_image.transpose(1, 2, 0), 0, 1).astype(np.float32)
        inpainted_display = np.clip(inpainted_image.transpose(1, 2, 0), 0, 1).astype(np.float32)
        mask_display = np.repeat(mask_array.transpose(1, 2, 0), 3, axis=2).astype(np.float32) / 255
        axes[0].imshow(original_display)
        axes[0].set_title("After Stage 1 (Small Gaps Filled)")
        axes[0].axis("off")
        axes[1].imshow(inpainted_display)
        axes[1].set_title("Stage 2: Inpainted Image")
        axes[1].axis("off")
        axes[2].imshow(mask_display)
        axes[2].set_title("Mask (White = Missing)")
        axes[2].axis("off")
        plt.suptitle(f"Trip {trip_id} - Two-Stage Gap Filling")
        plt.tight_layout()
        plt.show()

    return final_df, metrics


def generate_artificial_gap(
    trip_df: pd.DataFrame,
    gap_start_fraction: float = 0.75,
    gap_end_fraction: float = 0.95,
    h3_resolution: Optional[int] = None,
    rename_map: Optional[Mapping[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    trip_data = prepare_trip_df(
        trip_df,
        required_columns=TRIP_REQUIRED_COLUMNS,
        h3_resolution=h3_resolution,
        rename_map=rename_map,
    ).sort_values("time").reset_index(drop=True)
    if not 0 <= gap_start_fraction < gap_end_fraction <= 1:
        raise ValueError("Gap fractions must satisfy 0 <= start < end <= 1.")
    row_count = len(trip_data)
    gap_start_index = int(row_count * gap_start_fraction)
    gap_end_index = max(gap_start_index + 1, int(row_count * gap_end_fraction))
    gap_end_index = min(gap_end_index, row_count)
    trip_with_missing = trip_data.copy()
    trip_with_missing.loc[gap_start_index:gap_end_index - 1, ["lon", "lat", "h3"]] = np.nan
    metadata = {
        "gap_start_index": gap_start_index,
        "gap_end_index": gap_end_index,
        "gap_size": gap_end_index - gap_start_index,
    }
    return trip_with_missing, metadata


def compute_gap_dtw_metrics(
    original_gap_df: pd.DataFrame,
    filled_gap_df: pd.DataFrame,
) -> dict[str, float]:
    try:
        from dtaidistance import dtw, dtw_ndim
    except ImportError as exc:
        raise ImportError("dtaidistance is required to compute DTW metrics.") from exc

    original_gap = original_gap_df.copy()
    filled_gap = filled_gap_df.copy()
    gap_size = len(original_gap)
    if gap_size == 0:
        return {
            "gap_size": 0,
            "dtw_lon": float("nan"),
            "dtw_lat": float("nan"),
            "dtw_combined": float("nan"),
            "h3_match_count": 0,
            "h3_match_pct": float("nan"),
        }

    gt_lon = original_gap["lon"].values.astype(np.float64)
    gt_lat = original_gap["lat"].values.astype(np.float64)
    filled_lon = filled_gap["lon"].values.astype(np.float64)
    filled_lat = filled_gap["lat"].values.astype(np.float64)
    filled_lon_clean = pd.Series(filled_lon).ffill().bfill().values
    filled_lat_clean = pd.Series(filled_lat).ffill().bfill().values

    dtw_lon = float(dtw.distance(gt_lon, filled_lon_clean))
    dtw_lat = float(dtw.distance(gt_lat, filled_lat_clean))
    gt_2d = np.column_stack([gt_lon, gt_lat])
    filled_2d = np.column_stack([filled_lon_clean, filled_lat_clean])
    dtw_combined = float(dtw_ndim.distance(gt_2d, filled_2d))
    h3_match_count = int((original_gap["h3"].values == filled_gap["h3"].values).sum())
    h3_match_pct = h3_match_count / gap_size * 100 if gap_size > 0 else float("nan")

    return {
        "gap_size": gap_size,
        "dtw_lon": dtw_lon,
        "dtw_lat": dtw_lat,
        "dtw_combined": dtw_combined,
        "h3_match_count": h3_match_count,
        "h3_match_pct": h3_match_pct,
    }


def create_trip_folium_map(
    original_trip_df: pd.DataFrame,
    filled_trip_df: pd.DataFrame,
    gap_start_index: int,
    gap_end_index: int,
    output_path: str | Path,
    trip_id: Optional[str] = None,
) -> str:
    try:
        import folium
    except ImportError as exc:
        raise ImportError("folium is required to create HTML trajectory maps.") from exc

    original_coords = original_trip_df[["lat", "lon"]].dropna()
    if original_coords.empty:
        raise ValueError("Original trip does not contain any valid coordinates for map generation.")

    center_lat = float(original_coords["lat"].mean())
    center_lon = float(original_coords["lon"].mean())
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    original_points = original_trip_df[["lat", "lon"]].dropna().values.tolist()
    folium.PolyLine(
        original_points,
        weight=3,
        color="blue",
        opacity=0.8,
        popup="Original Trip (Ground Truth)",
    ).add_to(folium_map)

    if original_points:
        folium.Marker(
            original_points[0],
            popup="Original Start",
            icon=folium.Icon(color="blue", icon="play"),
        ).add_to(folium_map)
        folium.Marker(
            original_points[-1],
            popup="Original End",
            icon=folium.Icon(color="blue", icon="stop"),
        ).add_to(folium_map)

    filled_points = filled_trip_df[["lat", "lon"]].dropna().values.tolist()
    folium.PolyLine(
        filled_points,
        weight=2,
        color="red",
        opacity=0.6,
        dash_array="5, 10",
        popup="Filled Trip",
    ).add_to(folium_map)

    filled_gap_data = filled_trip_df.iloc[gap_start_index:gap_end_index][["lat", "lon"]].dropna()
    if len(filled_gap_data) > 0:
        gap_points = filled_gap_data.values.tolist()
        folium.PolyLine(
            gap_points,
            weight=4,
            color="green",
            opacity=0.9,
            popup=f"Filled Gap (indices {gap_start_index}-{gap_end_index})",
        ).add_to(folium_map)
        for point_index, (_, row) in enumerate(filled_gap_data.iterrows()):
            if point_index % 10 == 0:
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=4,
                    color="green",
                    fill=True,
                    fill_color="green",
                    fill_opacity=0.7,
                    popup=f"Filled point {point_index}",
                ).add_to(folium_map)

    legend_html = """
<div style=\"position: fixed; bottom: 50px; left: 50px; width: 200px; height: 120px;
            border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
            padding: 10px; border-radius: 5px;\">
<b>Legend</b><br>
<i style=\"background:blue; width:30px; height:3px; display:inline-block;\"></i> Original Trip<br>
<i style=\"background:red; width:30px; height:3px; display:inline-block; border-style:dashed;\"></i> Filled Trip<br>
<i style=\"background:green; width:30px; height:3px; display:inline-block;\"></i> Filled Gap Region<br>
<i style=\"background:green; width:10px; height:10px; display:inline-block; border-radius:50%;\"></i> Sample Points
</div>
"""
    folium_map.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(folium_map)

    output_file = _ensure_parent_dir(output_path)
    folium_map.save(output_file)
    if trip_id is not None:
        print(f"Saved folium map for trip {trip_id} to {output_file.as_posix()}")
    return output_file.as_posix()
