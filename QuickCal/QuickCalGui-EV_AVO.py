import copy
import io
import json
import os
import re
import warnings
from datetime import datetime

import gsw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from glob import glob
from matplotlib.patches import Circle
import matplotlib.patches as patches
from scipy import optimize
from scipy.signal import argrelextrema

# Ensure MACEFunctions are accessible
sys.path.insert(0, "G:/WPy64-31150/applications/MaceFunctions")

# Try imports - handle gracefully if running in environment without them for UI testing
try:
    import tsCalc
    import win32com.client
    from echolab2.instruments import echosounder
    from echolab2.plotting.matplotlib import echogram
    from echolab2.processing import line, grid, integration
    from matplotlib.pyplot import figure, show
except ImportError as e:
    print(f"Warning: Critical dependencies missing ({e}). GUI will load but calibration will fail.")

CONFIG_MODE_QUICKCAL = "quickcal"
CONFIG_MODE_AVO = "avo"

DEFAULT_AVO_SETTINGS = {
    "temp": 5.5,
    "salinity": 32.5,
    "lat": 55.0,
    "sphere_diameter": 38.1,
    "sphere_material": "Tungsten carbide",
    "beam_width_deg": 6.5,
    "vessel": "Dyson",
    "subsector_divisions": 6,
    "min_targets_per_division": 1,
    "sphere_depth_tolerance_m": 4.0,
}


TS_BANDWIDTH_PULSE_LENGTHS_US = [64, 128, 256, 512, 1024, 2048, 4096]

TS_BANDWIDTH_TABLE_KHZ = {
    18:  {64: np.nan, 128: np.nan, 256: np.nan, 512: 1.75,  1024: 1.57, 2048: 1.19, 4096: 0.72},
    38:  {64: np.nan, 128: np.nan, 256: 3.68,   512: 3.28,  1024: 2.43, 2048: 1.45, 4096: 0.77},
    70:  {64: 6.93,   128: 6.74,   256: 6.09,   512: 4.63,  1024: 2.83, 2048: 1.51, 4096: np.nan},
    120: {64: 11.66,  128: 10.79,  256: 8.61,   512: 5.49,  1024: 2.99, 2048: 1.53, 4096: np.nan},
    200: {64: 18.54,  128: 15.55,  256: 10.51,  512: 5.9,   1024: 3.05, 2048: np.nan, 4096: np.nan},
    333: {64: np.nan, 128: np.nan, 256: np.nan, 512: 1.96,  1024: 0.98, 2048: np.nan, 4096: np.nan},
}


def _ts_bandwidth_from_channel_name(channel_id, pulse_duration=None, frequency=None):
    """
    Return the configured TS bandwidth in Hz for a channel name and pulse duration.

    The bandwidth values (in kHz) are looked up from the combination of frequency
    and pulse duration (in microseconds). Where the table value is NaN (or for
    unsupported combinations), 1 / pulse_duration of the calibration object is used.

    Parameters
    ----------
    channel_id : str
        Channel identifier string (e.g. 'WBT 998500-15 ES18_1' or '38 kHz').
    pulse_duration : float or None
        Pulse duration in seconds (e.g. 0.001024) or microseconds (e.g. 1024).
    frequency : float or None
        Transducer frequency in Hz (e.g. 18000) or kHz (e.g. 18).

    Returns
    -------
    float
        Bandwidth in Hz.
    """
    # 1. Identify frequency (in kHz)
    channel_khz = None
    if frequency is not None:
        try:
            f_val = float(_scalar_at(frequency)) if hasattr(frequency, '__len__') or isinstance(frequency, np.ndarray) else float(frequency)
            if f_val > 1000:
                f_val /= 1000.0
            closest_f = min(TS_BANDWIDTH_TABLE_KHZ.keys(), key=lambda k: abs(k - f_val))
            if abs(closest_f - f_val) < 5.0:
                channel_khz = closest_f
        except Exception:
            pass

    if channel_khz is None:
        channel_name = str(channel_id)
        for khz in TS_BANDWIDTH_TABLE_KHZ.keys():
            pattern = rf"(?<!\d){khz}(?:\.0+)?\s*(?:k(?:hz)?)?(?!\d)"
            if re.search(pattern, channel_name, flags=re.IGNORECASE):
                channel_khz = khz
                break

    # 2. Identify pulse duration (seconds and microseconds)
    pulse_duration_sec = None
    pulse_len_us = None
    if pulse_duration is not None:
        try:
            p_val = float(_scalar_at(pulse_duration)) if hasattr(pulse_duration, '__len__') or isinstance(pulse_duration, np.ndarray) else float(pulse_duration)
            if p_val > 0:
                if p_val < 0.1:  # Value is in seconds
                    pulse_duration_sec = p_val
                    pulse_len_us = p_val * 1e6
                else:  # Value is in microseconds
                    pulse_len_us = p_val
                    pulse_duration_sec = p_val / 1e6
        except Exception:
            pass

    # 3. Lookup in table if frequency and pulse duration are recognized
    if channel_khz in TS_BANDWIDTH_TABLE_KHZ and pulse_len_us is not None:
        closest_col = min(TS_BANDWIDTH_PULSE_LENGTHS_US, key=lambda c: abs(c - pulse_len_us))
        if abs(closest_col - pulse_len_us) / closest_col <= 0.10:
            table_val = TS_BANDWIDTH_TABLE_KHZ[channel_khz].get(closest_col)
            if table_val is not None and not np.isnan(table_val):
                bw_hz = float(table_val) * 1000.0
                print(f"  > Using TS bandwidth: {table_val:.2f} kHz (table lookup) for {channel_khz} kHz @ {closest_col} us.")
                return bw_hz
            else:
                # Value is NaN -> use 1 / pulse duration
                fallback_bw = 1.0 / pulse_duration_sec
                print(f"  > Using TS bandwidth: {fallback_bw/1000.0:.3f} kHz (1/tau, table NaN) for {channel_khz} kHz @ {pulse_len_us:.0f} us.")
                return fallback_bw
        else:
            # Pulse length not close to any standard column -> fallback to 1 / pulse duration
            fallback_bw = 1.0 / pulse_duration_sec
            print(f"  > Using TS bandwidth: {fallback_bw/1000.0:.3f} kHz (1/tau, non-standard pulse) for {channel_khz} kHz @ {pulse_len_us:.0f} us.")
            return fallback_bw

    # 4. Fallback if pulse duration is provided but channel_khz is not in table
    if pulse_duration_sec is not None and pulse_duration_sec > 0:
        fallback_bw = 1.0 / pulse_duration_sec
        print(f"  > Using TS bandwidth: {fallback_bw/1000.0:.3f} kHz (1/tau) for channel {channel_id}.")
        return fallback_bw

    # 5. Fallback if pulse duration was not provided at all
    if channel_khz in TS_BANDWIDTH_TABLE_KHZ:
        val_default = TS_BANDWIDTH_TABLE_KHZ[channel_khz].get(1024)
        if val_default is None or np.isnan(val_default):
            val_default = TS_BANDWIDTH_TABLE_KHZ[channel_khz].get(512, 1.0)
        bw_hz = float(val_default) * 1000.0
        print(f"  > WARNING: Pulse duration not provided for {channel_id}; defaulting TS bandwidth to {val_default:.2f} kHz.")
        return bw_hz

    supported = ", ".join(f"{frequency} kHz" for frequency in TS_BANDWIDTH_TABLE_KHZ)
    raise ValueError(
        f"Could not determine TS bandwidth from channel name {channel_id!r}. "
        f"Expected one of: {supported}."
    )


def read_ctd_file(file_path):
    """
    Reads a CTD file (.cnv or .csv) and returns a pandas DataFrame
    with 'temp', 'sal', 'depth', and 'pressure' columns.
    """
    ext = file_path.split('.')[-1].lower()
    if ext == 'cnv':
        try:
            with open(file_path, 'r', encoding='utf-8') as file_handle:
                lines = file_handle.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file_handle:
                lines = file_handle.readlines()

        data_start_line = -1
        header_lines = []
        for index, line_text in enumerate(lines):
            if line_text.startswith('*END*'):
                data_start_line = index + 1
                header_lines = lines[:index]
                break

        if data_start_line == -1:
            raise ValueError("Could not find *END* of header in CNV file.")

        column_names = {}
        name_pattern = re.compile(r'# name (\d+) = (.*?):')
        for line_text in header_lines:
            match = name_pattern.match(line_text)
            if match:
                column_names[int(match.group(1))] = match.group(2).strip()

        if not column_names:
            raise ValueError("Could not parse column names from CNV file header.")

        data_io = io.StringIO(''.join(lines[data_start_line:]))
        data_frame = pd.read_csv(data_io, sep=r"\s+", header=None)

        if data_frame.shape[1] < len(column_names):
            data_frame = data_frame.rename(columns={k: v for k, v in column_names.items() if k < data_frame.shape[1]})
        else:
            data_frame = data_frame.rename(columns=column_names)

        temp_col = None
        sal_col = None
        pres_col = None
        depth_col = None
        svel_col = None

        for column_name in data_frame.columns:
            if isinstance(column_name, str):
                column_name_lower = column_name.lower()
                if 'temp' in column_name_lower or 't090c' in column_name_lower or 'tv290c' in column_name_lower:
                    temp_col = column_name
                if 'sal' in column_name_lower or 'sal00' in column_name_lower:
                    sal_col = column_name
                if 'pres' in column_name_lower or 'prdm' in column_name_lower:
                    pres_col = column_name
                if 'dep' in column_name_lower or 'depth' in column_name_lower:
                    depth_col = column_name
                if 'sv' in column_name_lower or 'svel' in column_name_lower or 'sound' in column_name_lower:
                    svel_col = column_name

        if not temp_col:
            raise ValueError("Could not find temperature column in CNV file.")
        if not sal_col:
            raise ValueError("Could not find salinity column in CNV file.")

        result_df = pd.DataFrame()
        return_cols = ['temp', 'sal', 'depth', 'pressure']

        result_df['temp'] = pd.to_numeric(data_frame[temp_col], errors='coerce')
        result_df['sal'] = pd.to_numeric(data_frame[sal_col], errors='coerce')

        if pres_col:
            result_df['pressure'] = pd.to_numeric(data_frame[pres_col], errors='coerce')
            result_df['depth'] = abs(gsw.z_from_p(result_df['pressure'], 55.0))
        elif depth_col:
            result_df['depth'] = pd.to_numeric(data_frame[depth_col], errors='coerce')
            result_df['pressure'] = gsw.p_from_z(-result_df['depth'], 55.0)
        else:
            raise ValueError("Could not find pressure or depth column in CNV file.")

        if svel_col:
            result_df['svel'] = pd.to_numeric(data_frame[svel_col], errors='coerce')
            return_cols.append('svel')

        result_df.dropna(inplace=True)
        return result_df[return_cols]

    if ext == 'csv':
        data_frame = pd.read_csv(file_path)
        data_frame.rename(
            columns={
                'Depth (Meter)': 'depth',
                'Temperature (Celsius)': 'temp',
                'Salinity (Practical Salinity Scale)': 'sal',
            },
            inplace=True,
        )

        required_cols = ['depth', 'temp', 'sal']
        if not all(column_name in data_frame.columns for column_name in required_cols):
            raise ValueError(
                "CSV file must contain 'Depth (Meter)', 'Temperature (Celsius)', "
                "and 'Salinity (Practical Salinity Scale)' columns."
            )

        if 'depth' in data_frame.columns and 'pressure' not in data_frame.columns:
            data_frame['pressure'] = gsw.p_from_z(-data_frame['depth'], 55.0)

        return data_frame[['temp', 'sal', 'pressure', 'depth']]

    raise ValueError(f"Unsupported CTD file extension: {ext}. Only .cnv and .csv are supported.")


def get_default_avo_settings():
    return copy.deepcopy(DEFAULT_AVO_SETTINGS)


def normalize_calibration_mode(value):
    return CONFIG_MODE_AVO if str(value).strip().lower() == CONFIG_MODE_AVO else CONFIG_MODE_QUICKCAL


def merge_avo_settings(settings):
    merged = get_default_avo_settings()
    if isinstance(settings, dict):
        merged.update(settings)
    return merged


def _yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_single_quoted(value):
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _format_inline_list(values, quoted=False):
    formatter = _yaml_single_quoted if quoted else _yaml_scalar
    return "[" + ", ".join(formatter(value) for value in values) + "]"


QUICKCAL_DETECTION_PARAMETER_COMMENTS = {
    "PLDL": "Pulse length determination level (dB below the pulse peak).",
    "maxNormPulseLen": "Maximum normalized pulse length allowed for accepted targets.",
    "minNormPulseLen": "Minimum normalized pulse length allowed for accepted targets.",
    "maxBeamComp": "Maximum beam compensation value allowed for accepted targets.",
    "maxSDalong": "Maximum alongship angle standard deviation for accepted targets.",
    "maxSDathwart": "Maximum athwartship angle standard deviation for accepted targets.",
    "min_threshold": "Minimum target strength (dB) retained during target detection.",
    "max_threshold": "Maximum target strength (dB) retained during target detection.",
}


AVO_SETTINGS_COMMENTS = {
    "temp": "Representative seawater temperature in degrees Celsius.",
    "salinity": "Representative seawater salinity in PSU.",
    "lat": "Latitude in decimal degrees used for seawater property calculations.",
    "sphere_diameter": "Calibration sphere diameter in millimeters.",
    "sphere_material": "Calibration sphere material name used by the AVO workflow.",
    "beam_width_deg": "Nominal transducer beam width in degrees.",
    "vessel": "Vessel name used to label AVO outputs.",
    "subsector_divisions": "Number of angular subsectors used for beam coverage checks.",
    "min_targets_per_division": "Minimum accepted targets required in each subsector.",
    "sphere_depth_tolerance_m": "Allowed +/- depth window around sphere_range in meters.",
}


def _format_block_mapping(mapping, indent=2):
    lines = []
    pad = " " * indent
    for key, value in mapping.items():
        lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
    return lines


def _append_comment(lines, comment, indent=0):
    lines.append(f'{" " * indent}# {comment}')


def _append_commented_scalar(lines, key, value, comment, indent=0):
    _append_comment(lines, comment, indent=indent)
    lines.append(f'{" " * indent}{key}: {_yaml_scalar(value)}')


def _append_commented_list_item_scalar(lines, key, value, comment, indent=2):
    _append_comment(lines, comment, indent=indent)
    lines.append(f'{" " * indent}- {key}: {_yaml_scalar(value)}')


def _append_commented_inline_list(lines, key, values, comment, indent=0, quoted=False):
    _append_comment(lines, comment, indent=indent)
    lines.append(f'{" " * indent}{key}: {_format_inline_list(values or [], quoted=quoted)}')


ENVIRONMENT_SETTINGS_COMMENTS = {
    "manual_env": "Whether to use manual environmental settings instead of CTD profile.",
    "manual_temp": "Manual temperature in degrees Celsius.",
    "manual_sal": "Manual salinity in PSU.",
    "manual_c": "Manual sound speed in m/s.",
    "transducer_depth": "Transducer depth in meters.",
}


def _append_commented_block_mapping(lines, key, mapping, header_comment, field_comments=None, indent=0):
    if field_comments is None:
        field_comments = {}
    _append_comment(lines, header_comment, indent=indent)
    lines.append(f'{" " * indent}{key}:')
    for field_key, field_value in (mapping or {}).items():
        _append_commented_scalar(
            lines,
            field_key,
            field_value,
            field_comments.get(field_key, f"Value for {field_key}."),
            indent=indent + 2,
        )


def _append_commented_block_list(lines, key, values, comment, indent=0):
    _append_comment(lines, comment, indent=indent)
    lines.append(f'{" " * indent}{key}:')
    for value in values or []:
        lines.append(f'{" " * (indent + 2)}- {_yaml_scalar(value)}')


def _format_channel_block(channel, include_avo_settings):
    if include_avo_settings:
        lines = []
        _append_commented_list_item_scalar(lines, "id", channel.get("id", ""), "Channel ID.", indent=2)
        _append_commented_block_list(
            lines,
            "raw_files",
            channel.get("raw_files", []),
            "Raw file(s) or wildcard patterns used for this channel.",
            indent=4,
        )
        _append_commented_scalar(
            lines,
            "sphere_range",
            channel.get("sphere_range", ""),
            "Sphere range in meters during on-axis collection.",
            indent=4,
        )
        return lines

    lines = []
    _append_commented_list_item_scalar(lines, "id", channel.get("id", ""), "Channel ID.", indent=2)
    _append_commented_inline_list(
        lines,
        "raw_files",
        channel.get("raw_files", []),
        "Raw file(s). Use wildcards like [.../*.raw] to capture all files in a folder.",
        indent=4,
        quoted=True,
    )
    _append_commented_scalar(
        lines,
        "sphere_range",
        channel.get("sphere_range", ""),
        "Sphere range in meters during on-axis collection.",
        indent=4,
    )

    if channel.get("sphere_size") not in (None, ""):
        _append_commented_scalar(lines, "sphere_size", channel.get("sphere_size"), "Sphere size in millimeters.", indent=4)
    if channel.get("sphere_material"):
        _append_commented_scalar(
            lines,
            "sphere_material",
            channel.get("sphere_material"),
            "Sphere material used to calculate the reference target strength.",
            indent=4,
        )
    if channel.get("sphere_ts_tolerance") not in (None, ""):
        _append_commented_scalar(
            lines,
            "sphere_ts_tolerance",
            channel.get("sphere_ts_tolerance"),
            "Allowed +/- target strength window around the calculated reference TS.",
            indent=4,
        )
    if channel.get("min_ts") not in (None, ""):
        _append_commented_scalar(lines, "min_ts", channel.get("min_ts"), "Explicit minimum target strength override in dB.", indent=4)
    if channel.get("max_ts") not in (None, ""):
        _append_commented_scalar(lines, "max_ts", channel.get("max_ts"), "Explicit maximum target strength override in dB.", indent=4)
    if channel.get("bad_data_regions_file"):
        _append_commented_scalar(
            lines,
            "bad_data_regions_file",
            channel.get("bad_data_regions_file"),
            "Channel-specific bad data regions file (.evr) override.",
            indent=4,
        )
    det_params = channel.get("detection_parameters") or {}
    if det_params:
        _append_commented_block_mapping(
            lines,
            "detection_parameters",
            det_params,
            "Channel-specific single target detection parameter overrides.",
            QUICKCAL_DETECTION_PARAMETER_COMMENTS,
            indent=4,
        )
    return lines


def format_saved_config_yaml(config):
    if normalize_calibration_mode(config.get("calibration_mode")) == CONFIG_MODE_AVO:
        lines = [
            "##### Configuration file for QuickCalGui_AVO in AVO mode #####",
            "",
        ]
        _append_commented_scalar(
            lines,
            "calibration_mode",
            CONFIG_MODE_AVO,
            'Set to "avo" to run the integrated AVO coverage-check workflow.',
        )
        lines.extend([
            "",
            "### Output definitions ###",
            "",
        ])
        _append_commented_scalar(
            lines,
            "output_directory",
            config.get("output_directory", ""),
            "Base folder where AVO figures and summary files will be written.",
        )
        lines.append("")
        _append_commented_scalar(
            lines,
            "default_ctd",
            config.get("default_ctd", ""),
            "Optional CTD file used to calculate the average sound speed along the transducer-to-sphere path.",
        )
        lines.append("")
        _append_commented_block_mapping(
            lines,
            "environment_settings",
            config.get("environment_settings", {}),
            "Transducer depth used as the start of the CTD sound-speed averaging interval.",
            {"transducer_depth": "Transducer depth in meters."},
        )
        lines.extend([
            "",
            "### Hidden AVO settings migrated from AVO.ini ###",
            "",
        ])
        _append_commented_block_mapping(
            lines,
            "avo_settings",
            merge_avo_settings(config.get("avo_settings", {})),
            "AVO-only settings migrated from AVO.ini.",
            AVO_SETTINGS_COMMENTS,
        )
        lines.extend([
            "",
            "### Channel definitions ###",
            "",
            "channels:",
        ])
        channels = config.get("channels", []) or []
        if channels:
            for channel in channels:
                lines.extend(_format_channel_block(channel, include_avo_settings=True))
                lines.append("")
            lines.pop()
        return "\n".join(lines) + "\n"

    lines = [
        "##### Configuration file for QuickCal tool #####",
        "",
        "# Before using this file, set the CTD file, channel sphere ranges, and raw file paths.",
        "",
        "### Global definitions ###",
        "",
    ]
    _append_commented_scalar(
        lines,
        "output_directory",
        config.get("output_directory", ""),
        "Output directory for calibration results.",
    )
    lines.append("")
    _append_commented_scalar(
        lines,
        "default_ctd",
        config.get("default_ctd", ""),
        "Full path to the CTD file (Sea-Bird .cnv or CastAway .csv).",
    )
    lines.append("")
    if config.get("bad_data_regions_file"):
        _append_commented_scalar(
            lines,
            "bad_data_regions_file",
            config.get("bad_data_regions_file", ""),
            "Optional bad data regions file (.evr) to exclude corrupted or noisy time windows.",
        )
        lines.append("")
    _append_commented_scalar(
        lines,
        "default_sphere_size",
        config.get("default_sphere_size", 38.1),
        "Default sphere size in millimeters; channel-specific values override this.",
    )
    _append_commented_scalar(
        lines,
        "default_sphere_material",
        config.get("default_sphere_material", "Tungsten carbide"),
        "Default sphere material; channel-specific values override this.",
    )
    _append_commented_scalar(
        lines,
        "sphere_range_tolerance",
        config.get("sphere_range_tolerance", 1),
        "Allowed +/- range around each channel's sphere_range, in meters.",
    )
    _append_commented_scalar(
        lines,
        "sphere_ts_tolerance",
        config.get("sphere_ts_tolerance", 1),
        "Allowed +/- target strength window around the calculated reference TS.",
    )
    _append_commented_block_mapping(
        lines,
        "environment_settings",
        config.get("environment_settings", {}),
        "Environment values and transducer depth used for CTD averaging.",
    )
    lines.append("")
    _append_commented_block_mapping(
        lines,
        "detection_parameters",
        config.get("detection_parameters", {}),
        "Global single target detection parameters; channel-specific overrides win.",
        QUICKCAL_DETECTION_PARAMETER_COMMENTS,
    )
    lines.extend([
        "",
        "### Channel definitions and parameters ###",
        "",
        "channels:",
    ])
    channels = config.get("channels", []) or []
    if channels:
        for channel in channels:
            lines.extend(_format_channel_block(channel, include_avo_settings=False))
            lines.append("")
        lines.pop()
    return "\n".join(lines) + "\n"


def expand_raw_files(files):
    if not isinstance(files, list):
        files = [files]

    expanded_files = []
    for file_path in files:
        matches = glob(file_path)
        if matches:
            expanded_files.extend(matches)
        else:
            expanded_files.append(file_path)

    return sorted(list(set(expanded_files)))

# ==============================================================================
#  ORIGINAL CORE CLASSES
# ==============================================================================

class detectParmsInit():
    """Detection parameters initialized from a configuration dictionary."""
    def __init__(self, config=None):
        config = config or {}
        self.PLDL = config.get('PLDL', 6)
        self.maxNormPulseLen = config.get('maxNormPulseLen', 20)
        self.minNormPulseLen = config.get('minNormPulseLen', 0.1)
        self.maxBeamComp = config.get('maxBeamComp', 0.1)
        self.maxSDalong = config.get('maxSDalong', 0.6)
        self.maxSDathwart = config.get('maxSDathwart', 0.6)
        self.excludeBelow = config.get('excludeBelow', 1e10)
        self.excludeAbove = config.get('excludeAbove', 0)
        self.min_threshold = config.get('min_threshold', -50)
        self.max_threshold = config.get('max_threshold', -20)


def _scalar_at(value, index=0):
    """Return a calibration value as a scalar, supporting scalar/array values."""
    values = np.asarray(value)
    return float(values if values.ndim == 0 else values[index])


def _resolve_sound_speed_and_density(
    ctd_file,
    transducer_depth,
    sphere_range,
    calibration_sound_speed,
    latitude,
    fallback_temp=10.0,
    fallback_salinity=35.0,
):
    """Resolve the common sound speed and density used by a calibration run."""
    cal_sound_speed = _scalar_at(calibration_sound_speed)
    sphere_depth = float(transducer_depth) + float(sphere_range)

    if ctd_file and os.path.exists(ctd_file):
        ctd_df = read_ctd_file(ctd_file).reset_index(drop=True)
        path_df = ctd_df[
            (ctd_df["depth"] >= float(transducer_depth))
            & (ctd_df["depth"] <= sphere_depth)
        ].copy()
        if path_df.empty:
            raise ValueError(
                f"No CTD data found between {transducer_depth}m and {sphere_depth:.2f}m."
            )

        if "svel" in path_df.columns:
            sound_speed = 1 / np.mean(1 / path_df["svel"].to_numpy())
        else:
            path_c, path_rho = tsCalc.water_properties(
                path_df["sal"].values,
                path_df["temp"].values,
                path_df["pressure"].values,
                lon=0.0,
                lat=latitude,
            )
            sound_speed = 1 / np.mean(1 / path_c)
            return float(sound_speed), float(np.mean(path_rho))

        _, path_rho = tsCalc.water_properties(
            path_df["sal"].values,
            path_df["temp"].values,
            path_df["pressure"].values,
            lon=0.0,
            lat=latitude,
        )
        return float(sound_speed), float(np.mean(path_rho))

    _, fallback_rho = tsCalc.water_properties(
        np.array([fallback_salinity]),
        np.array([fallback_temp]),
        np.array([sphere_depth]),
        lon=0.0,
        lat=latitude,
    )
    return cal_sound_speed, float(np.asarray(fallback_rho).flat[0])


def _integrate_sphere_window(
    data,
    center_range,
    sound_speed,
    pulse_duration,
    layer_axis,
    half_width_pulse_lengths=1.0,
):
    """Integrate the sphere echo over a window centered on the echo envelope."""
    # Single-target detection calculates range as:
    #   target_range = (envelope centroid) - (sound_speed * pulse_duration / 4)
    # To center the integration window symmetrically on the physical echo envelope,
    # we add (sound_speed * pulse_duration / 4) to center_range.
    pulse_term = sound_speed * pulse_duration / 4.0
    echo_center = center_range + pulse_term
    integration_half_width = (sound_speed * pulse_duration / 2.0) * half_width_pulse_lengths
    print(integration_half_width)
    upper_line = line.line(
        ping_time=data.ping_time,
        data=echo_center - integration_half_width,
    )
    lower_line = line.line(
        ping_time=data.ping_time,
        data=echo_center + integration_half_width,
    )

    integrator = integration.integrator(min_threshold_applied=False)
    grid_obj = grid.grid(
        interval_length=10000,
        interval_axis="ping_number",
        data=data,
        layer_axis=layer_axis,
        layer_thickness=100,
    )
    return integrator.integrate(
        data,
        grid_obj,
        exclude_above_line=upper_line,
        exclude_below_line=lower_line,
    )


def _reject_overlapping_candidates(candidates):
    """Apply Method 2 overlap rejection, retaining the stronger target."""
    accepted = []
    for candidate in sorted(candidates, key=lambda item: item['r']):
        overlapping = [
            item for item in accepted
            if item['envelope_start'] <= candidate['envelope_end']
            and candidate['envelope_start'] <= item['envelope_end']
        ]
        if not overlapping:
            accepted.append(candidate)
            continue

        if all(candidate['cTS'] > item['cTS'] for item in overlapping):
            accepted = [item for item in accepted if item not in overlapping]
            accepted.append(candidate)

    return sorted(accepted, key=lambda item: item['r'])


def _calculate_pulse_width(power, peak_index, limit, range_vector, sound_speed, pulse_duration):
    """Calculate the Echoview-style normalized pulse-envelope width."""
    right_indices = np.where(power[peak_index:] < limit)[0]
    left_indices = np.where(power[:peak_index] < limit)[0]
    if right_indices.size == 0 or left_indices.size == 0:
        return None

    right = peak_index + right_indices[0] - 1
    left = left_indices[-1] + 1
    x_left = left + (limit - power[left]) / (power[left + 1] - power[left])
    x_right = right + (limit - power[right]) / (power[right + 1] - power[right])
    sample_indices = np.arange(len(range_vector))
    envelope_start = np.interp(x_left, sample_indices, range_vector)
    envelope_end = np.interp(x_right, sample_indices, range_vector)
    normalized_width = (envelope_end - envelope_start) / (sound_speed * pulse_duration / 2)
    return normalized_width, left, right, envelope_start, envelope_end


def _detect_single_target_candidates(
    d_sp,
    cal,
    along,
    athwart,
    ping,
    params,
    target_range_min=None,
    target_range_max=None,
    sound_speed_override=None,
):
    """Return Method 2 single-target candidates for one ping."""
    abs_coeff = _scalar_at(cal.absorption_coefficient, ping)
    full_range_vector = d_sp.range
    sound_speed = (
        _scalar_at(sound_speed_override)
        if sound_speed_override is not None
        else _scalar_at(cal.sound_speed, ping)
    )
    pulse_duration = _scalar_at(cal.pulse_duration, ping)
    pulse_term = sound_speed * pulse_duration / 4

    if target_range_min is None and target_range_max is None:
        range_start = 0
        range_end = len(full_range_vector)
    else:
        range_step = np.median(np.diff(full_range_vector)) if len(full_range_vector) > 1 else 0.0
        envelope_padding = max(sound_speed * pulse_duration, 2 * range_step)
        lower_bound = -np.inf if target_range_min is None else target_range_min
        upper_bound = np.inf if target_range_max is None else target_range_max
        search_mask = (
            (full_range_vector >= lower_bound - envelope_padding)
            & (full_range_vector <= upper_bound + envelope_padding)
        )
        matching_indices = np.flatnonzero(search_mask)
        if matching_indices.size == 0:
            return []
        range_start = int(matching_indices[0])
        range_end = int(matching_indices[-1]) + 1

    range_vector = full_range_vector[range_start:range_end]
    compensated_range_term = 40 * np.log10(range_vector) + 2 * abs_coeff * range_vector
    calibrated_power = d_sp.data[ping][range_start:range_end] - compensated_range_term
    maxima = argrelextrema(calibrated_power, np.greater)[0]
    candidates = []

    minimum_threshold = params.min_threshold if hasattr(params, 'min_threshold') else params.threshold_min
    maximum_threshold = params.max_threshold if hasattr(params, 'max_threshold') else params.threshold_max

    for peak_index in maxima:
        pldl_value = calibrated_power[peak_index] - params.PLDL
        result = _calculate_pulse_width(
            calibrated_power,
            peak_index,
            pldl_value,
            range_vector,
            sound_speed,
            pulse_duration,
        )
        if result is None:
            continue

        normalized_width, left, right, envelope_start, envelope_end = result
        if normalized_width > params.maxNormPulseLen or normalized_width < params.minNormPulseLen:
            continue

        peak_along = along.data[ping][range_start + peak_index]
        peak_athwart = athwart.data[ping][range_start + peak_index]
        along_norm = 2 * peak_along / cal.beam_width_alongship[ping]
        athwart_norm = 2 * peak_athwart / cal.beam_width_athwartship[ping]
        beam_compensation = 6.0206 * (
            along_norm ** 2
            + athwart_norm ** 2
            - (0.18 * along_norm ** 2 * athwart_norm ** 2)
        )
        if beam_compensation > params.maxBeamComp:
            continue

        start_index = left + 1
        end_index = right - 1
        if (end_index - start_index) < 1:
            continue
        angle_start = range_start + start_index
        angle_end = range_start + end_index
        if np.std(along.data[ping][angle_start:angle_end]) > params.maxSDalong:
            continue
        if np.std(athwart.data[ping][angle_start:angle_end]) > params.maxSDathwart:
            continue

        envelope_power = 10 ** (calibrated_power[left:right + 1] / 10)
        target_range = (
            np.sum(range_vector[left:right + 1] * envelope_power)
            / np.sum(envelope_power)
            - pulse_term
        )
        if not np.isfinite(target_range) or target_range <= 0:
            continue
        if target_range > params.excludeBelow or target_range < params.excludeAbove:
            continue

        uncompensated_ts = (
            calibrated_power[peak_index]
            + 40 * np.log10(target_range)
            + 2 * abs_coeff * target_range
        )
        compensated_ts = uncompensated_ts + beam_compensation
        if not minimum_threshold <= compensated_ts <= maximum_threshold:
            continue

        candidates.append({
            'r': target_range,
            'uTS': uncompensated_ts,
            'cTS': compensated_ts,
            'peakAthwart': peak_athwart,
            'peakAlong': peak_along,
            'sdAlng': np.std(along.data[ping][angle_start:angle_end]),
            'sdAthw': np.std(athwart.data[ping][angle_start:angle_end]),
            'normWidth': normalized_width,
            'envelope_start': envelope_start - pulse_term,
            'envelope_end': envelope_end - pulse_term,
        })

    return _reject_overlapping_candidates(candidates)

class singleTargetsInit():
    """Container for detected single target attributes with subsetting capability."""
    def __init__(self):
        self.ping = np.array([])
        self.r = np.array([])
        self.uTS = np.array([])
        self.cTS = np.array([])
        self.peakAthwart = np.array([])
        self.peakAlong = np.array([])
        self.normWidth = np.array([])
   
    def get_subset(self, indices):
        """Returns a new instance containing only the specified indices."""
        subset = singleTargetsInit()
        for attr, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(subset, attr, value[indices])
        return subset

class EchosounderCalibration:
    def __init__(
        self,
        channel_id,
        raw_files,
        ctd_file,
        env_settings=None,
        bad_data_regions_file=None,
        sphere_size=38.1,
        sphere_mat='Tungsten carbide',
        sphere_range=21.0,
        detect_config=None,
    ):
        self.channel_id = channel_id
        self.raw_files = self._build_file_list(raw_files)
        self.ctd_file = ctd_file
        self.env_settings = env_settings or {}
        self.bad_data_regions_file = bad_data_regions_file
        self.sphere_size = sphere_size
        self.sphere_mat = sphere_mat
        self.sphere_range = sphere_range
        
        self.ek_data = None
        self.cal = None
        self.along = None
        self.athwart = None
        self.lat = None
        self.lon = None
        self.d_sv = None
        self.d_sp = None
        self.sound_speed = None
        self.water_density = None
        
        self.params = detectParmsInit(detect_config)
        self.targets = singleTargetsInit()
        self.sphere_targets = None
        self.bad_data_regions = []

    def _load_bad_data_regions(self):
        """Loads and parses a bad data regions EVR file."""
        if not self.bad_data_regions_file or not os.path.exists(self.bad_data_regions_file):
            return

        print(f"  > Loading bad data regions from: {os.path.basename(self.bad_data_regions_file)}")
        try:
            with open(self.bad_data_regions_file, 'r', encoding='utf-8-sig') as file_handle:
                lines = [line.strip() for line in file_handle.readlines()]

            if not lines or not lines[0].startswith('EVRG'):
                print("  - ERROR: Not a valid EVR file (missing EVRG header).")
                if lines and "," in lines[0]:
                    print("  - INFO: This appears to be a CSV file. Please use the new .evr format.")
                return

            num_regions = int(lines[1])
            line_idx = 2

            for _ in range(num_regions):
                if line_idx >= len(lines):
                    break

                if lines[line_idx] == '':
                    line_idx += 1

                if line_idx >= len(lines):
                    break
                region_header = lines[line_idx].split()
                line_idx += 1

                point_count = int(region_header[1])

                if line_idx >= len(lines):
                    break
                num_notes = int(lines[line_idx])
                line_idx += 1 + num_notes

                if line_idx >= len(lines):
                    break
                num_detection_settings = int(lines[line_idx])
                line_idx += 1 + num_detection_settings

                if line_idx >= len(lines):
                    break

                while line_idx < len(lines):
                    parts = lines[line_idx].split()
                    if len(parts) > 5 and parts[0].isdigit():
                        break
                    line_idx += 1

                if line_idx >= len(lines):
                    break
                points_line = lines[line_idx].split()
                line_idx += 1

                region_type = 0
                if line_idx < len(lines) and lines[line_idx].isdigit():
                    region_type = int(lines[line_idx])
                    line_idx += 1

                if line_idx < len(lines):
                    line_idx += 1

                if region_type not in [0, 4]:
                    continue

                if point_count == 4 and len(points_line) >= 9:
                    start_date = points_line[0]
                    start_time_raw = points_line[1]
                    end_date = points_line[6]
                    end_time_raw = points_line[7]

                    start_time_str = start_time_raw.ljust(10, '0')
                    start_dt = datetime.strptime(
                        f"{start_date}{start_time_str[:6]}",
                        '%Y%m%d%H%M%S',
                    ).replace(microsecond=int(start_time_str[6:10]) * 100)

                    end_time_str = end_time_raw.ljust(10, '0')
                    end_dt = datetime.strptime(
                        f"{end_date}{end_time_str[:6]}",
                        '%Y%m%d%H%M%S',
                    ).replace(microsecond=int(end_time_str[6:10]) * 100)

                    self.bad_data_regions.append((start_dt, end_dt))

            print(f"  > Found {len(self.bad_data_regions)} bad data time regions to exclude.")

        except Exception as exc:
            print(f"  - ERROR: Failed to read or process bad data regions file: {exc}")
            import traceback
            traceback.print_exc()

    def _build_file_list(self, files):
        """Expands wildcards and ensures a sorted list of unique files."""
        return expand_raw_files(files)

    def load_data(self):
        """Reads raw data and extracts calibration/angle information."""
        print(f"Loading {len(self.raw_files)} files for channel {self.channel_id}...")
        self.ek_data = echosounder.read(self.raw_files, channel_ids=[self.channel_id])
        self.cal = echosounder.get_calibration_from_raw(self.ek_data)[self.channel_id]
        
        chan_obj = self.ek_data.get_channel_data()[self.channel_id][0]
        self.along, self.athwart = chan_obj.get_physical_angles(calibration=self.cal)
        
        try:
            gga = chan_obj.nmea_data.get_datagrams('GGA')['GGA']['data'][0]
            if self.lat is None:
                self.lat = float(gga.lat[:2]) + (float(gga.lat[2:]) / 60)
            self.lon = float(gga.lon[:3]) + (float(gga.lon[3:]) / 60)
        except (IndexError, KeyError):
            print("  > Warning: Could not find GGA datagram in raw file. Using default latitude of 55.0 N.")
            if self.lat is None:
                self.lat = 55.0
            if self.lon is None:
                self.lon = 0.0
        
        self.d_sv = echosounder.get_Sv(self.ek_data)[self.channel_id]
        self.d_sp = echosounder.get_Sp(self.ek_data)[self.channel_id]
        self._load_bad_data_regions()

    def get_reference_ts(self):
        transducer_depth = self.env_settings.get('transducer_depth', 9.15)
        sound_speed, rho = _resolve_sound_speed_and_density(
            self.ctd_file,
            transducer_depth,
            self.sphere_range,
            self.cal.sound_speed,
            self.lat,
            fallback_temp=self.env_settings.get('manual_temp', 10.0),
            fallback_salinity=self.env_settings.get('manual_sal', 35.0),
        )
        self.sound_speed = sound_speed
        self.water_density = rho
        if self.ctd_file and os.path.exists(self.ctd_file):
            print(
                f"  > Using average CTD sound speed from {transducer_depth}m "
                f"to {transducer_depth + self.sphere_range:.2f}m: {sound_speed:.2f} m/s."
            )
        else:
            print(f"  > Using calibration-object sound speed: {sound_speed:.2f} m/s.")

        material = tsCalc.material_properties()[self.sphere_mat]

        cal_sound_speed = np.mean(self.cal.sound_speed)
        if round(sound_speed, 1) != round(cal_sound_speed, 1):
            print(
                f"  > WARNING: Sound speed mismatch. Resolved value is {sound_speed:.1f} m/s, "
                f"while calibration object value is {cal_sound_speed:.1f} m/s."
            )
        
        f = _scalar_at(self.cal.frequency)
        pulse_duration = _scalar_at(self.cal.pulse_duration)
        bw = _ts_bandwidth_from_channel_name(self.channel_id, pulse_duration=pulse_duration, frequency=f)
        
        fr, ts = tsCalc.freq_response(f-bw/2, f+bw/2, self.sphere_size/1000/2, sound_speed, 
                                      material['c1'], material['c2'], rho, material['rho1'], fstep=100)
        return 10 * np.log10(np.mean(10**(ts/10)))

    def detect_targets(self, target_range_min=None, target_range_max=None):
        if not self.bad_data_regions and self.bad_data_regions_file:
            self._load_bad_data_regions()
        ping_times_dt = pd.to_datetime(self.d_sp.ping_time)
        self.targets = singleTargetsInit()

        for ping in range(self.d_sp.n_pings):
            ping_time = ping_times_dt[ping]
            if any(start_time <= ping_time <= end_time for start_time, end_time in self.bad_data_regions):
                continue
            for candidate in _detect_single_target_candidates(
                self.d_sp,
                self.cal,
                self.along,
                self.athwart,
                ping,
                self.params,
                target_range_min=target_range_min,
                target_range_max=target_range_max,
                sound_speed_override=self.sound_speed,
            ):
                self._append_target(
                    ping, candidate['r'], candidate['uTS'], candidate['cTS'],
                    candidate['peakAthwart'], candidate['peakAlong'], candidate['normWidth'],
                )

    def _calculate_pulse_width(self, power, l, limit, range_vector, sound_speed, pulse_duration):
        r_idx = np.where(power[l:] < limit)[0]
        l_idx = np.where(power[:l] < limit)[0]
        if r_idx.size == 0 or l_idx.size == 0: return None
        
        right, left = l + r_idx[0] - 1, l_idx[-1] + 1
        xLeft = left + (limit - power[left]) / (power[left+1] - power[left])
        xRight = right + (limit - power[right]) / (power[right+1] - power[right])
        sample_indices = np.arange(len(range_vector))
        envelope_start = np.interp(xLeft, sample_indices, range_vector)
        envelope_end = np.interp(xRight, sample_indices, range_vector)
        normalized_width = (envelope_end - envelope_start) / (sound_speed * pulse_duration / 2)
        return normalized_width, left, right, envelope_start, envelope_end

    def _get_beam_comp(self, ping, p_along, p_athwart):
        al = 2 * p_along / self.cal.beam_width_alongship[ping]
        at = 2 * p_athwart / self.cal.beam_width_athwartship[ping]
        return 6.0206 * (al**2 + at**2 - (0.18 * al**2 * at**2))

    def _check_stdev(self, ping, start, end):
        if (end - start) < 1: return False
        if np.std(self.along.data[ping][start:end]) > self.params.maxSDalong: return False
        if np.std(self.athwart.data[ping][start:end]) > self.params.maxSDathwart: return False
        return True

    def _append_target(self, ping, r, uTS, cTS, athw, alng, width):
        self.targets.ping = np.append(self.targets.ping, ping)
        self.targets.r = np.append(self.targets.r, r)
        self.targets.uTS = np.append(self.targets.uTS, uTS)
        self.targets.cTS = np.append(self.targets.cTS, cTS)
        self.targets.peakAthwart = np.append(self.targets.peakAthwart, athw)
        self.targets.peakAlong = np.append(self.targets.peakAlong, alng)
        self.targets.normWidth = np.append(self.targets.normWidth, width)

    def run_calibration(self, range_tolerance=1.0, ts_tolerance=1.0, min_ts=None, max_ts=None, plot=True, plot_save_dir=None):
        print(f"Running calibration for {self.channel_id} with {self.sphere_size} mm sphere" )
        ref_ts = self.get_reference_ts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.detect_targets(
                target_range_min=self.sphere_range - range_tolerance,
                target_range_max=self.sphere_range + range_tolerance,
            )
        
        # Determine TS Mask: If explicit min/max TS provided, use them. Else use tolerance around Ref TS.
        if min_ts is not None and max_ts is not None:
             ts_mask = (self.targets.cTS >= min_ts) & (self.targets.cTS <= max_ts)
             print(f"  > Using explicit TS range: {min_ts} to {max_ts} dB")
        else:
             ts_mask = (np.abs(self.targets.cTS - ref_ts) < ts_tolerance)
             print(f"  > Using TS tolerance: +/- {ts_tolerance} dB around {ref_ts:.2f} dB")

        mask = (np.abs(self.targets.r - self.sphere_range) < range_tolerance) & ts_mask
        sphere_hits = np.where(mask)
        self.sphere_targets = self.targets.get_subset(sphere_hits)
        
        if len(self.sphere_targets.ping) == 0:
            print(f"Error: No sphere targets detected for {self.channel_id}.")
            return None

        d_sv_on_axis = self.d_sv.copy()
        target_pings = np.unique(self.sphere_targets.ping).astype(int)
        all_pings = np.arange(self.d_sv.n_pings)
        pings_no_sphere = all_pings[~np.isin(all_pings, target_pings)]
        
        d_sv_on_axis.delete(index_array=pings_no_sphere)
        d_sv_on_axis.range = self.d_sv.range

        maximum_plot_range = self.sphere_range + 25.0
        range_indices = np.where(self.d_sv.range <= maximum_plot_range)[0]
        if range_indices.size > 0:
            d_sv_on_axis = d_sv_on_axis.view(
                (0, -1, 1),
                (0, int(range_indices[-1]), 1),
            )

        observed_ts = 10 * np.log10(np.mean(10**(self.sphere_targets.cTS / 10)))
        observed_ts_std = np.std(self.sphere_targets.cTS)
        mean_range = np.mean(self.sphere_targets.r)

        if plot:
            clean_id = self.channel_id.replace(' ', '_').replace('-', '_').replace(':', '_')+'-'+str(self.sphere_size).split('.')[0]
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(self.channel_id)
            ax[0].hist(self.sphere_targets.cTS, bins=100)
            ax[0].set_title(f"TS Distribution (N={len(self.sphere_targets.cTS)})")
            ax[1].plot(self.sphere_targets.peakAlong, self.sphere_targets.peakAthwart, '.')
            ax[1].set_title(f"Beam Positions: {self.channel_id}")
            ax[2].plot(self.sphere_targets.ping, self.sphere_targets.r, '.')
            ax[2].set_title("Single Target Range")
            ax[2].set_xlabel("Single Target Number")
            ax[2].set_ylabel("Range (m)")
            ax[2].grid(True)
            ax[2].invert_yaxis()
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            if plot_save_dir:
                plt.savefig(os.path.join(plot_save_dir, f"{clean_id}_stats.png"))
            plt.close(fig)

            fig_echo = figure(figsize=(12, 4))
            eg = echogram.Echogram(fig_echo, d_sv_on_axis, threshold=[-90, -30])
            pulse_duration = _scalar_at(self.cal.pulse_duration)
            pulse_term = self.sound_speed * pulse_duration / 4.0
            hw = (self.sound_speed * pulse_duration / 2.0)
            u_line = line.line(ping_time=d_sv_on_axis.ping_time, data=mean_range + pulse_term - hw)
            l_line = line.line(ping_time=d_sv_on_axis.ping_time, data=mean_range + pulse_term + hw)
            eg.plot_line(u_line, color='black', linewidth=1, linestyle='--')
            eg.plot_line(l_line, color='black', linewidth=1, linestyle='--')
            eg.add_colorbar(fig_echo)
            plt.title(f"Echogram: {self.channel_id}")
            if plot_save_dir:
                plt.savefig(os.path.join(plot_save_dir, f"{clean_id}_echogram.png"))
            plt.close(fig_echo)

        integrated = _integrate_sphere_window(
            d_sv_on_axis,
            mean_range,
            self.sound_speed,
            _scalar_at(self.cal.pulse_duration),
            layer_axis="range",
        )

        eba = np.unique(self.cal.equivalent_beam_angle)[0]
        ref_nasc = (10**(ref_ts / 10) * (1852**2) * 4 * np.pi) / ((10**(eba / 10)) * (mean_range**2))
        obs_nasc = integrated.nasc[0][0]
        used_gain = np.unique(self.cal.gain + self.cal.sa_correction)[0]
        new_sv_gain = np.unique((self.cal.gain + self.cal.sa_correction) - (10 * np.log10(ref_nasc / obs_nasc)) / 2)[0]
        calc_gain = (((observed_ts - ref_ts) / 2) + np.unique(self.cal.gain))[0]
        sa_corr = new_sv_gain - calc_gain

        return {
            'channel_id': self.channel_id,
            'sphere_size':self.sphere_size,
            'observed_ts': observed_ts,
            'observed_ts_std': observed_ts_std,
            'reference_ts': ref_ts,
            'target_range_m': mean_range,
            'observed_nasc': obs_nasc,
            'reference_nasc': ref_nasc,
            'ping_count': len(d_sv_on_axis.ping_time),
            'calc_ts_gain': calc_gain,
            'sa_correction': sa_corr,
            'new_sv_gain': new_sv_gain,
            'files_used': ", ".join(self.raw_files)
        }

def run_batch_calibration(config_path, do_plot=True):
    """
    Original Function logic maintained.
    Takes a path to a yaml config, processes it, and saves results.
    """
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    base_dir = config.get('output_directory', './cal_results')
    base_dir = os.path.join(base_dir, 'calibration_output')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    global_ctd = config.get('default_ctd')
    env_settings = config.get('environment_settings', {})
    global_sphere = config.get('default_sphere_size', 38.1)
    global_sphere_mat = config.get('default_sphere_material', 'Tungsten carbide')
    global_range_tol = config.get('sphere_range_tolerance', 1.0)
    global_ts_tol = config.get('sphere_ts_tolerance', 1.0)
    global_detect_params = config.get('detection_parameters', {})
    bad_data_regions_file = config.get('bad_data_regions_file')

    all_results = []
    if 'channels' in config and config['channels']:
        for ch_conf in config['channels']:
            channel_id = ch_conf['id']
            ch_detect = global_detect_params.copy()
            ch_detect.update(ch_conf.get('detection_parameters', {}))
            
            # Extract new Channel Specific TS parameters (with global fallback for tolerance)
            ch_ts_tol = ch_conf.get('sphere_ts_tolerance', global_ts_tol)
            ch_min_ts = ch_conf.get('min_ts', None)
            ch_max_ts = ch_conf.get('max_ts', None)

            try:
                cal_session = EchosounderCalibration(
                    channel_id=channel_id,
                    raw_files=ch_conf['raw_files'],
                    ctd_file=ch_conf.get('ctd_file', global_ctd),
                    env_settings=env_settings,
                    bad_data_regions_file=ch_conf.get('bad_data_regions_file', bad_data_regions_file),
                    sphere_range=ch_conf['sphere_range'],
                    sphere_size=ch_conf.get('sphere_size', global_sphere),
                    sphere_mat=ch_conf.get('sphere_material', global_sphere_mat),
                    detect_config=ch_detect
                )
                cal_session.load_data()
                res = cal_session.run_calibration(
                    range_tolerance=ch_conf.get('sphere_range_tolerance', global_range_tol),
                    ts_tolerance=ch_ts_tol,
                    min_ts=ch_min_ts,
                    max_ts=ch_max_ts,
                    plot=do_plot, plot_save_dir=plots_dir
                )
                if res: all_results.append(res)
            except Exception as e:
                print(f"FAILED {channel_id}: {e}")
                # Print stack trace for debugging via GUI
                import traceback
                traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("D%m%d%y-T%H%M%S")
        csv_path = os.path.join(base_dir, f"CalibrationSummary_{timestamp}.csv")
        yaml_path = os.path.join(base_dir, f"CalibrationSummary_{timestamp}.yml")
        df.to_csv(csv_path, index=False)
        print(f"Complete, see {csv_path} for results")

        try:
            with open(yaml_path, 'w') as file_handle:
                yaml.safe_dump(config, file_handle, sort_keys=False, default_flow_style=False)
            print(f"Configuration saved to {yaml_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    else:
        print("No results generated.")


class AVODetectParams:
    def __init__(self):
        self.PLDL = 6
        self.maxNormPulseLen = 20
        self.minNormPulseLen = 0.1
        self.maxBeamComp = 6
        self.maxSDalong = 2
        self.maxSDathwart = 2
        self.excludeBelow = 1e10
        self.excludeAbove = 0
        self.threshold_min = -55
        self.threshold_max = -30


class AVOSingleTargets:
    def __init__(self):
        self.ping = np.array([])
        self.r = np.array([])
        self.uTS = np.array([])
        self.cTS = np.array([])
        self.peakAthwart = np.array([])
        self.peakAlong = np.array([])
        self.sdAlng = np.array([])
        self.sdAthw = np.array([])
        self.normWidth = np.array([])

    def get_subset(self, indices):
        subset = AVOSingleTargets()
        for attr, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(subset, attr, value[indices])
        return subset


class TriwaveCorrect:
    def __init__(self, start_sample, end_sample):
        self.start_sample = start_sample
        self.end_sample = end_sample

    def triwave_correct(self, data_in):
        sample_count = data_in.n_samples
        ringdown = np.log10(np.mean(10 ** (data_in.power[:, self.start_sample:self.end_sample]), axis=1))

        nan_indices = np.argwhere(np.isnan(ringdown))
        while np.any(nan_indices):
            ringdown[nan_indices] = ringdown[nan_indices - 1]
            nan_indices = np.argwhere(np.isnan(ringdown))

        inf_indices = np.argwhere(np.isinf(ringdown))
        if len(ringdown) != len(inf_indices):
            while np.any(inf_indices):
                ringdown[inf_indices] = ringdown[inf_indices - 1]
                inf_indices = np.argwhere(np.isinf(ringdown))

        fit_results = self.fit_triangle(ringdown)
        generated_triangle_offset = self.general_triangle(
            np.arange(data_in.shape[0]),
            A=fit_results["amplitude"],
            M=2721.0,
            k=fit_results["period_offset"],
            C=0,
            dtype="float32",
        )
        triangle_matrix_correct = np.array([generated_triangle_offset] * sample_count).transpose()
        data_in.power = data_in.power - triangle_matrix_correct

        return data_in, fit_results, True

    def fit_triangle(self, mean_ringdown_vec, amplitude=None, period_offset=None, amplitude_offset=None):
        sample_indices = np.arange(len(mean_ringdown_vec))
        fit_func = lambda params: self.general_triangle(sample_indices, params[0], 2721.0, params[1], params[2])
        err_func = lambda params: (mean_ringdown_vec - fit_func(params))

        if period_offset is None:
            period_offset = 1360 - np.argmax(mean_ringdown_vec)
        if amplitude is None:
            amplitude = 1.0
        if amplitude_offset is None:
            amplitude_offset = np.mean(mean_ringdown_vec)

        fit_params, fit_cov, fit_info, fit_msg, fit_success = optimize.leastsq(
            err_func,
            [amplitude, period_offset, amplitude_offset],
            full_output=True,
        )

        ss_total = sum((mean_ringdown_vec - mean_ringdown_vec.mean()) ** 2)
        ss_error = sum(err_func(fit_params) ** 2)
        fit_r_squared = 1 - ss_error / ss_total

        fit_amplitude, fit_period_offset, fit_amplitude_offset = fit_params
        if fit_amplitude < 0:
            fit_amplitude = -fit_amplitude
            fit_period_offset += 2721.0 / 2
        fit_period_offset = fit_period_offset % 2721

        if abs(fit_period_offset - 2721) < abs(fit_period_offset):
            fit_period_offset -= 2721

        return {
            "period_offset": fit_period_offset,
            "amplitude_offset": fit_amplitude_offset,
            "amplitude": fit_amplitude,
            "r_squared": fit_r_squared,
        }

    def general_triangle(self, sample_indices, A=0.5, M=2721, k=0, C=0, dtype=None):
        phase = ((sample_indices + k) % M) / float(M)
        triangle = A * (2 * abs(2 * (phase - np.floor(phase + 0.5))) - 1) + C
        if dtype is not None:
            return triangle.astype(dtype)
        return triangle


class AVOCalibrationSession:
    def __init__(self, channel_id, raw_files, sphere_range, settings, output_dir, ctd_file=None, transducer_depth=9.15):
        self.channel_id = channel_id
        self.raw_files = expand_raw_files(raw_files)
        self.sphere_range = float(sphere_range)
        self.settings = merge_avo_settings(settings)
        self.ctd_file = ctd_file
        self.transducer_depth = float(transducer_depth)
        self.output_dir = output_dir
        self.figure_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.figure_dir, exist_ok=True)

        self.params = AVODetectParams()
        self.targets = AVOSingleTargets()
        self.sphere_targets = None
        self.ek_data = None
        self.channel_data = None
        self.cal = None
        self.along = None
        self.athwart = None
        self.d_sv = None
        self.d_sp = None
        self.lat = None
        self.frequency = None
        self.pulse_length_us = None
        self.sphere_depth = None
        self.ping_day = None
        self.triwave_corrected = False
        self.triwave_fit_results = None
        self.sound_speed = None
        self.water_density = None

    def _first_scalar(self, value):
        arr = np.atleast_1d(value)
        if arr.size == 0:
            raise ValueError(f"No value available for {self.channel_id}")
        return float(arr[0])

    def _safe_filename(self):
        return self.channel_id.replace(" ", "_").replace("-", "_").replace(":", "_")

    def load_data(self):
        print(f"Loading {len(self.raw_files)} files for AVO check: {self.channel_id}")
        self.ek_data = echosounder.read(self.raw_files, channel_ids=[self.channel_id])
        self.cal = echosounder.get_calibration_from_raw(self.ek_data)[self.channel_id]
        self.channel_data = self.ek_data.get_channel_data()[self.channel_id][0]

        if self._requires_triwave_correction():
            print("  > GPT/ES80 data detected, applying triangle wave correction.")
            self.channel_data, self.triwave_fit_results, _ = TriwaveCorrect(0, 5).triwave_correct(self.channel_data)
            self.triwave_corrected = True

        self.along, self.athwart = self.channel_data.get_physical_angles(calibration=self.cal)
        self.d_sv = self.channel_data.get_Sv(calibration=self.cal)
        self.d_sv.to_depth()
        self.d_sp = self.channel_data.get_Sp(calibration=self.cal)

        self.frequency = int(np.round(self._first_scalar(getattr(self.channel_data, "frequency", self.cal.frequency))))
        self.pulse_length_us = int(np.round(self._first_scalar(self.cal.pulse_duration) * 1e6))
        self.ping_day = np.datetime_as_string(self.d_sv.ping_time[0], unit="D")
        self.sphere_depth = self.sphere_range + float(self.d_sv.depth[0])

        try:
            gga = self.channel_data.nmea_data.get_datagrams("GGA")["GGA"]["data"][0]
            self.lat = float(gga.lat[:2]) + (float(gga.lat[2:]) / 60)
        except Exception:
            self.lat = float(self.settings["lat"])
            print(f"  > No GPS latitude found; using AVO default latitude {self.lat}.")

        if self.triwave_fit_results is not None:
            fit_path = os.path.join(
                self.figure_dir,
                f"TriangleCorrection-{self.settings['vessel']}-{self.frequency}-{self.ping_day}.txt",
            )
            with open(fit_path, "w") as fit_file:
                print(self.triwave_fit_results, file=fit_file)

    def _requires_triwave_correction(self):
        configuration = self.channel_data.configuration[0]
        return (
            configuration.get("transceiver_type") == "GPT"
            and configuration.get("application_name") == "ES80"
        )

    def get_reference_ts(self):
        if hasattr(self.channel_data, "is_cw") and not self.channel_data.is_cw():
            raise ValueError("AVO mode only supports CW data.")

        material = tsCalc.material_properties()[self.settings["sphere_material"]]
        sound_speed, density = _resolve_sound_speed_and_density(
            self.ctd_file,
            self.transducer_depth,
            self.sphere_range,
            self.cal.sound_speed,
            self.lat,
            fallback_temp=self.settings["temp"],
            fallback_salinity=self.settings["salinity"],
        )
        self.sound_speed = sound_speed
        self.water_density = density
        if self.ctd_file and os.path.exists(self.ctd_file):
            print(
                f"  > Using average CTD sound speed from {self.transducer_depth}m "
                f"to {self.transducer_depth + self.sphere_range:.2f}m: {sound_speed:.2f} m/s."
            )
        pulse_duration = self._first_scalar(self.cal.pulse_duration)
        bandwidth = _ts_bandwidth_from_channel_name(
            self.channel_id,
            pulse_duration=pulse_duration,
            frequency=self.frequency,
        )
        _, ts = tsCalc.freq_response(
            self.frequency - bandwidth / 2,
            self.frequency + bandwidth / 2,
            float(self.settings["sphere_diameter"]) / 1000 / 2,
            sound_speed,
            material["c1"],
            material["c2"],
            density,
            material["rho1"],
            fstep=100,
        )
        return 10 * np.log10(np.mean(10 ** (ts / 10)))

    def detect_targets(self):
        self.targets = AVOSingleTargets()
        for ping in range(self.d_sp.n_pings):
            for candidate in _detect_single_target_candidates(
                self.d_sp,
                self.cal,
                self.along,
                self.athwart,
                ping,
                self.params,
                target_range_min=self.params.excludeAbove,
                target_range_max=self.params.excludeBelow,
                sound_speed_override=self.sound_speed,
            ):
                self.targets.ping = np.append(self.targets.ping, ping)
                self.targets.r = np.append(self.targets.r, candidate['r'])
                self.targets.uTS = np.append(self.targets.uTS, candidate['uTS'])
                self.targets.cTS = np.append(self.targets.cTS, candidate['cTS'])
                self.targets.peakAthwart = np.append(self.targets.peakAthwart, candidate['peakAthwart'])
                self.targets.peakAlong = np.append(self.targets.peakAlong, candidate['peakAlong'])
                self.targets.sdAlng = np.append(self.targets.sdAlng, candidate['sdAlng'])
                self.targets.sdAthw = np.append(self.targets.sdAthw, candidate['sdAthw'])
                self.targets.normWidth = np.append(self.targets.normWidth, candidate['normWidth'])

    def save_echograms(self):
        clean_id = self._safe_filename()
        vessel = self.settings["vessel"]

        # Keep the saved echogram focused on the calibration area.  The sphere
        # depth includes the transducer depth, so this is 25 m beyond the
        # configured sphere range in the echogram's depth coordinates.
        maximum_plot_depth = self.sphere_depth + 25.0
        full_indices = np.where(self.d_sv.depth <= maximum_plot_depth)[0]
        if full_indices.size > 0:
            d_sv_full = self.d_sv.view((0, -1, 1), (0, int(full_indices[-1]), 1))
        else:
            d_sv_full = self.d_sv

        fig_full = figure(figsize=(12, 9))
        full_echogram = echogram.Echogram(fig_full, d_sv_full, threshold=[-90, -30])
        full_echogram.add_colorbar(fig_full)
        plt.savefig(os.path.join(self.figure_dir, f"Echogram-{vessel}-{clean_id}-{self.ping_day}.png"))
        plt.close(fig_full)

        sphere_window = np.where(
            np.abs(self.d_sv.depth - self.sphere_depth) < float(self.settings["sphere_depth_tolerance_m"])
        )[0]
        if sphere_window.size > 1:
            d_sv_zoom = self.d_sv.view((0, -1, 1), (int(sphere_window[0]), int(sphere_window[-1]), 1))
            fig_zoom = figure(figsize=(12, 3))
            zoom_echogram = echogram.Echogram(fig_zoom, d_sv_zoom, threshold=[-90, -30])
            zoom_echogram.add_colorbar(fig_zoom)
            plt.savefig(os.path.join(self.figure_dir, f"EchogramZoom-{vessel}-{clean_id}-{self.ping_day}.png"))
            plt.close(fig_zoom)

    @staticmethod
    def calculate_distances(range_meters, angle_athwart_deg, angle_along_deg):
        angle_athwart_rad = np.radians(angle_athwart_deg)
        angle_along_rad = np.radians(angle_along_deg)
        return range_meters * np.tan(angle_athwart_rad), range_meters * np.tan(angle_along_rad)

    @staticmethod
    def find_points_in_sector(x_coords, y_coords, radius, sector_num=1):
        x_vals = np.array(x_coords)
        y_vals = np.array(y_coords)
        distances = np.sqrt(x_vals ** 2 + y_vals ** 2)
        angles_deg = (np.degrees(np.arctan2(y_vals, x_vals)) + 360) % 360
        sector_start = (sector_num - 1) * 45
        sector_end = sector_num * 45
        radius_mask = distances <= radius
        angle_mask = (angles_deg >= sector_start) & (angles_deg < sector_end)
        combined_mask = radius_mask & angle_mask
        return combined_mask, x_vals[combined_mask], y_vals[combined_mask]

    def create_beam_plot(self):
        beam_radius_rad = np.radians(float(self.settings["beam_width_deg"]) / 2.0)
        beam_radius_m = np.mean(self.sphere_targets.r) * np.tan(beam_radius_rad)
        athwart_dist, along_dist = self.calculate_distances(
            np.mean(self.sphere_targets.r),
            self.sphere_targets.peakAthwart,
            self.sphere_targets.peakAlong,
        )

        subsector_divisions = int(self.settings["subsector_divisions"])
        min_targets_per_division = float(self.settings["min_targets_per_division"])
        sector_is_complete = []

        plt.figure(figsize=(10, 7))
        radial_bins = np.linspace(0, beam_radius_m, subsector_divisions + 1)
        if beam_radius_m <= 0 or len(np.unique(radial_bins)) < 2:
            radial_bins = np.array([0, 1])

        for sector_num in range(1, 9):
            mask, sector_x, sector_y = self.find_points_in_sector(
                athwart_dist,
                along_dist,
                beam_radius_m,
                sector_num=sector_num,
            )
            counts = np.histogram(np.sqrt(sector_x ** 2 + sector_y ** 2), bins=radial_bins)[0]
            is_complete = not (counts < min_targets_per_division).any()
            sector_is_complete.append(is_complete)
            color = "darkgreen" if is_complete else "red"
            plt.scatter(sector_x, sector_y, 5, color=color, alpha=0.5)
            sector_start = (sector_num - 1) * 45
            sector_end = sector_num * 45
            sector_patch = patches.Wedge((0, 0), beam_radius_m, sector_start, sector_end, alpha=0.1, color=color)
            plt.gca().add_patch(sector_patch)

        on_axis_threshold_deg = float(self.settings["beam_width_deg"]) * 0.025
        num_on_axis = len(
            np.where(
                (np.abs(self.sphere_targets.peakAthwart) < on_axis_threshold_deg)
                & (np.abs(self.sphere_targets.peakAlong) < on_axis_threshold_deg)
            )[0]
        )
        if num_on_axis < 250:
            axis_color = "red"
        elif num_on_axis < 500:
            axis_color = "yellow"
        else:
            axis_color = "green"

        axis_circle = Circle((0, 0), beam_radius_m / 10, color=axis_color, linestyle="-", linewidth=1)
        beam_circle = Circle((0, 0), beam_radius_m, fill=False, color="k", linestyle="-", linewidth=2)
        plt.gca().add_patch(axis_circle)
        plt.gca().add_patch(beam_circle)
        plt.axis("off")
        plt.grid()
        plt.legend(
            handles=[
                patches.Patch(color="red", label="Low coverage in sector"),
                patches.Patch(color="yellow", label="Some coverage but not enough\n(on-axis only)"),
                patches.Patch(color="green", label="Good coverage"),
            ],
            bbox_to_anchor=(1, 0.6),
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.figure_dir,
                f"TargetsInBeam-{self.settings['vessel']}-{self._safe_filename()}-{self.ping_day}.png",
            )
        )
        plt.close()

        if not all(sector_is_complete):
            coverage_status = "low_sector_coverage"
        elif num_on_axis < 250:
            coverage_status = "low_on_axis_coverage"
        elif num_on_axis < 500:
            coverage_status = "partial_on_axis_coverage"
        else:
            coverage_status = "good"

        return beam_radius_m, num_on_axis, coverage_status

    def create_ts_histogram(self, ref_ts):
        fig = plt.figure(figsize=(8, 3.5))
        plt.subplot(111)
        histogram = plt.hist(self.sphere_targets.cTS, bins=100)
        plt.title("Single target detections in the specified range\nRed region is expected sphere TS")
        plt.fill_betweenx([0, np.max(histogram[0]) * 1.05], ref_ts - 1.5, ref_ts + 1.5, color="red", alpha=0.5)
        plt.ylim(0, np.max(histogram[0]) * 1.05)
        plt.grid()
        plt.savefig(
            os.path.join(
                self.figure_dir,
                f"TargetTS-{self.settings['vessel']}-{self._safe_filename()}-{self.ping_day}.png",
            )
        )
        plt.close(fig)

    def run_check(self, do_plot=True):
        self.load_data()
        ref_ts = self.get_reference_ts()
        self.params.excludeAbove = self.sphere_range - float(self.settings["sphere_depth_tolerance_m"])
        self.params.excludeBelow = self.sphere_range + float(self.settings["sphere_depth_tolerance_m"])

        if do_plot:
            self.save_echograms()

        print(f"Running AVO coverage check for {self.channel_id} at {self.frequency / 1000:.0f} kHz")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.detect_targets()

        sphere_hits = np.where(
            np.abs(self.targets.r - self.sphere_range) < (float(self.settings["sphere_depth_tolerance_m"]) / 2.0)
        )
        self.sphere_targets = self.targets.get_subset(sphere_hits)

        result = {
            "channel_id": self.channel_id,
            "frequency_hz": self.frequency,
            "sphere_range_m": self.sphere_range,
            "reference_ts": ref_ts,
            "sphere_hit_count": int(len(self.sphere_targets.ping)),
            "on_axis_hit_count": 0,
            "coverage_status": "no_sphere_hits",
            "triwave_corrected": self.triwave_corrected,
            "files_used": ", ".join(self.raw_files),
        }

        if len(self.sphere_targets.ping) == 0:
            print(f"  > No sphere hits were detected for {self.channel_id}.")
            return result

        result["mean_detected_range_m"] = float(np.mean(self.sphere_targets.r))
        result["mean_detected_ts_db"] = float(np.mean(self.sphere_targets.cTS))
        result["ts_std_db"] = float(np.std(self.sphere_targets.cTS))

        # Integrate only the pings containing sphere detections. AVO data is
        # represented in depth coordinates, so convert the detected mean range
        # to the corresponding mean depth before constructing the bounds.
        d_sv_on_axis = self.d_sv.copy()
        target_pings = np.unique(self.sphere_targets.ping).astype(int)
        all_pings = np.arange(self.d_sv.n_pings)
        d_sv_on_axis.delete(index_array=all_pings[~np.isin(all_pings, target_pings)])
        mean_sphere_depth = result["mean_detected_range_m"] + float(self.d_sv.depth[0])
        integrated = _integrate_sphere_window(
            d_sv_on_axis,
            mean_sphere_depth,
            self.sound_speed,
            _scalar_at(self.cal.pulse_duration),
            layer_axis="depth",
        )
        result["observed_nasc"] = float(integrated.nasc[0][0])

        if do_plot:
            beam_radius_m, on_axis_hit_count, coverage_status = self.create_beam_plot()
            self.create_ts_histogram(ref_ts)
        else:
            beam_radius_m = np.mean(self.sphere_targets.r) * np.tan(np.radians(float(self.settings["beam_width_deg"]) / 2.0))
            on_axis_threshold_deg = float(self.settings["beam_width_deg"]) * 0.025
            on_axis_hit_count = len(
                np.where(
                    (np.abs(self.sphere_targets.peakAthwart) < on_axis_threshold_deg)
                    & (np.abs(self.sphere_targets.peakAlong) < on_axis_threshold_deg)
                )[0]
            )
            coverage_status = "good"

        result["beam_radius_m"] = float(beam_radius_m)
        result["on_axis_hit_count"] = int(on_axis_hit_count)
        result["coverage_status"] = coverage_status
        print(
            f"  > {self.channel_id}: {result['sphere_hit_count']} sphere hits, "
            f"{result['on_axis_hit_count']} on-axis hits, status={coverage_status}"
        )
        return result


def run_avo_batch_calibration(config_path, do_plot=True):
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file) or {}

    base_dir = os.path.join(config.get("output_directory", "./cal_results"), "avo_output")
    os.makedirs(base_dir, exist_ok=True)

    avo_settings = merge_avo_settings(config.get("avo_settings", {}))
    ctd_file = config.get("default_ctd")
    environment_settings = config.get("environment_settings", {})
    transducer_depth = environment_settings.get("transducer_depth", 9.15)
    all_results = []
    channels = config.get("channels", []) or []

    for channel in channels:
        channel_id = channel.get("id")
        if not channel_id:
            print("FAILED <unknown>: channel id is required.")
            continue

        try:
            session = AVOCalibrationSession(
                channel_id=channel_id,
                raw_files=channel["raw_files"],
                sphere_range=channel["sphere_range"],
                settings=avo_settings,
                output_dir=base_dir,
                ctd_file=channel.get("ctd_file", ctd_file),
                transducer_depth=transducer_depth,
            )
            all_results.append(session.run_check(do_plot=do_plot))
        except Exception as exc:
            print(f"FAILED {channel_id}: {exc}")
            import traceback
            traceback.print_exc()
            all_results.append(
                {
                    "channel_id": channel_id,
                    "coverage_status": "failed",
                    "error": str(exc),
                    "files_used": ", ".join(channel.get("raw_files", [])),
                }
            )

    if all_results:
        summary_path = os.path.join(base_dir, "AVOCheckSummary.csv")
        pd.DataFrame(all_results).to_csv(summary_path, index=False)
        print(f"Complete, see {summary_path} for AVO results.")
    else:
        print("No AVO results generated.")

# ==============================================================================
#  GUI CLASSES
# ==============================================================================

class TextRedirector(object):
    """Redirects stdout/stderr to a tkinter text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
    
    def flush(self):
        pass

class QuickCalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickCal v2 EV / AVO - Echosounder Calibration")
        self.root.geometry("900x800")
        self.calibration_mode = CONFIG_MODE_QUICKCAL
        self.avo_settings = get_default_avo_settings()
        self.mode_label_var = tk.StringVar(value="Mode: QuickCal")
        self.quickcal_only_widgets = []
        self.calibration_running = False

        # Variables for Global Settings
        self.output_dir = tk.StringVar()
        self.ctd_file = tk.StringVar()
        self.sphere_size = tk.DoubleVar(value=38.1)
        self.sphere_mat = tk.StringVar(value="Tungsten carbide")
        self.range_tol = tk.DoubleVar(value=2.0)
        self.ts_tol = tk.DoubleVar(value=1.0)
        self.bad_data_regions_file = tk.StringVar()

        self.manual_env = tk.BooleanVar(value=False)
        self.manual_temp = tk.DoubleVar(value=10.0)
        self.manual_sal = tk.DoubleVar(value=35.0)
        self.manual_c = tk.DoubleVar(value=1500.0)
        self.transducer_depth = tk.DoubleVar(value=9.15)
        
        # Detection Parameters
        self.det_PLDL = tk.DoubleVar(value=6)
        self.det_maxNormPulseLen = tk.DoubleVar(value=20)
        self.det_minNormPulseLen = tk.DoubleVar(value=0.1)
        self.det_maxBeamComp = tk.DoubleVar(value=0.1)
        self.det_maxSDalong = tk.DoubleVar(value=0.6)
        self.det_maxSDathwart = tk.DoubleVar(value=0.6)
        self.det_min_thresh = tk.DoubleVar(value=-50)
        self.det_max_thresh = tk.DoubleVar(value=-20)

        # Channels Data
        self.channels = [] # List of dictionaries

        self.create_widgets()
        self.apply_mode_state()
        self.update_env_status_label()

    def create_widgets(self):
        # Top Toolbar (Load/Save)
        top_frame = ttk.Frame(self.root, padding="5")
        top_frame.pack(fill=tk.X)
        ttk.Button(top_frame, text="Load Config (YAML)", command=self.load_yaml).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Save Config (YAML)", command=self.save_yaml_as).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, textvariable=self.mode_label_var).pack(side=tk.LEFT, padx=10)
        self.bad_data_button = ttk.Button(
            top_frame,
            text="Generate Bad Data Regions",
            command=self.generate_bad_data_regions_file,
        )
        self.bad_data_button.pack(side=tk.RIGHT, padx=5)
        self.quickcal_only_widgets.append(self.bad_data_button)
        self.run_button = ttk.Button(
            top_frame,
            text="RUN CALIBRATION",
            command=self.run_calibration_thread,
        )
        self.run_button.pack(side=tk.RIGHT, padx=5)

        # Main Scrollable Area
        main_scroll_container = ttk.Frame(self.root)
        main_scroll_container.pack(fill="both", expand=True)
        main_scroll_container.grid_rowconfigure(0, weight=1)
        main_scroll_container.grid_columnconfigure(0, weight=1)

        main_canvas = tk.Canvas(main_scroll_container)
        scrollbar = ttk.Scrollbar(main_scroll_container, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        scrollable_window = main_canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw"
        )
        main_canvas.bind(
            "<Configure>",
            lambda e: main_canvas.itemconfigure(scrollable_window, width=e.width)
        )
        main_canvas.configure(yscrollcommand=scrollbar.set)
        main_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # --- Section 1: Global Definitions ---
        f1 = ttk.LabelFrame(scrollable_frame, text="Global Definitions", padding="10")
        f1.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_file_entry(f1, "Output Directory:", self.output_dir, True)

        env_frame = ttk.Frame(f1)
        env_frame.pack(fill=tk.X, pady=2)
        ttk.Label(env_frame, text="Environmental Data:", width=20, anchor="e").pack(side=tk.LEFT)
        env_button = ttk.Button(env_frame, text="Calculate and Set Environment", command=self.open_env_dialog)
        env_button.pack(side=tk.LEFT, padx=5)
        self.env_status_label = ttk.Label(env_frame, text="CTD not set", foreground="red")
        self.env_status_label.pack(side=tk.LEFT, padx=5)
        self.quickcal_only_widgets.append(env_button)

        bad_data_frame = ttk.Frame(f1)
        bad_data_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bad_data_frame, text="Bad Data Regions File:", width=20, anchor="e").pack(side=tk.LEFT)
        bad_data_entry = ttk.Entry(bad_data_frame, textvariable=self.bad_data_regions_file)
        bad_data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        bad_data_load_button = ttk.Button(
            bad_data_frame,
            text="Load bad data regions",
            command=self.load_bad_data_regions,
        )
        bad_data_load_button.pack(side=tk.LEFT)
        bad_data_clear_button = ttk.Button(
            bad_data_frame,
            text="Clear",
            command=lambda: self.bad_data_regions_file.set(""),
        )
        bad_data_clear_button.pack(side=tk.LEFT, padx=(2, 0))
        self.quickcal_only_widgets.extend([bad_data_entry, bad_data_load_button, bad_data_clear_button])
        
        grid_f1 = ttk.Frame(f1)
        grid_f1.pack(fill=tk.X, pady=5)
        ttk.Label(grid_f1, text="Default Sphere Size (mm):").grid(row=0, column=0, sticky="e")
        sphere_size_entry = ttk.Entry(grid_f1, textvariable=self.sphere_size, width=10)
        sphere_size_entry.grid(row=0, column=1, sticky="w", padx=5)
        self.quickcal_only_widgets.append(sphere_size_entry)
        
        ttk.Label(grid_f1, text="Sphere Material:").grid(row=0, column=2, sticky="e")
        sphere_mat_entry = ttk.Entry(grid_f1, textvariable=self.sphere_mat, width=20)
        sphere_mat_entry.grid(row=0, column=3, sticky="w", padx=5)
        self.quickcal_only_widgets.append(sphere_mat_entry)

        ttk.Label(grid_f1, text="Range Tolerance (m):").grid(row=1, column=0, sticky="e")
        range_tol_entry = ttk.Entry(grid_f1, textvariable=self.range_tol, width=10)
        range_tol_entry.grid(row=1, column=1, sticky="w", padx=5)
        self.quickcal_only_widgets.append(range_tol_entry)

        ttk.Label(grid_f1, text="TS Tolerance (dB):").grid(row=1, column=2, sticky="e")
        ts_tol_entry = ttk.Entry(grid_f1, textvariable=self.ts_tol, width=10)
        ts_tol_entry.grid(row=1, column=3, sticky="w", padx=5)
        self.quickcal_only_widgets.append(ts_tol_entry)

        # --- Section 2: Global Detection Parameters ---
        f2 = ttk.LabelFrame(scrollable_frame, text="Global Detection Parameters", padding="10")
        f2.pack(fill=tk.X, padx=10, pady=5)
        
        params = [
            ("PLDL", self.det_PLDL),
            ("Max Norm Pulse Len", self.det_maxNormPulseLen),
            ("Min Norm Pulse Len", self.det_minNormPulseLen),
            ("Max Beam Comp", self.det_maxBeamComp),
            ("Max SD Along", self.det_maxSDalong),
            ("Max SD Athwart", self.det_maxSDathwart),
            ("Min Threshold", self.det_min_thresh),
            ("Max Threshold", self.det_max_thresh),
        ]
        
        for i, (label, var) in enumerate(params):
            r, c = divmod(i, 4)
            ttk.Label(f2, text=label+":").grid(row=r, column=c*2, sticky="e", padx=5, pady=2)
            entry = ttk.Entry(f2, textvariable=var, width=8)
            entry.grid(row=r, column=c*2+1, sticky="w", padx=5, pady=2)
            self.quickcal_only_widgets.append(entry)

        # --- Section 3: Channels ---
        f3 = ttk.LabelFrame(scrollable_frame, text="Channels (Double-click to edit)", padding="10")
        f3.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        btn_frame = ttk.Frame(f3)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Add Channel", command=lambda: self.open_channel_popup(None)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Edit Selected", command=self.edit_selected_channel).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_channel).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_channels).pack(side=tk.LEFT, padx=2)

        self.channel_listbox = tk.Listbox(f3, height=8)
        self.channel_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.channel_listbox.bind("<Double-Button-1>", lambda event: self.edit_selected_channel())

        
        # --- Section 4: Console Output ---
        f4 = ttk.LabelFrame(scrollable_frame, text="Console Output", padding="5")
        f4.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.console = scrolledtext.ScrolledText(f4, height=10, state="disabled")
        self.console.pack(fill="both", expand=True)

        # Redirect stdout/stderr
        sys.stdout = TextRedirector(self.console, "stdout")
        sys.stderr = TextRedirector(self.console, "stderr")

    def _keep_child_window_in_front(self, window):
        """Keep an editing window above the main GUI until it is closed."""
        window.transient(self.root)
        window.lift()
        window.focus_force()
        window.grab_set()

    def open_env_dialog(self):
        env_popup = tk.Toplevel(self.root)
        env_popup.title("Calculate and Set Environment")
        env_popup.geometry("500x350")
        self._keep_child_window_in_front(env_popup)

        top_frame = ttk.Frame(env_popup, padding=10)
        top_frame.pack(fill=tk.X)

        ctd_frame = ttk.LabelFrame(env_popup, text="CTD-based Calculation", padding=10)
        manual_frame = ttk.LabelFrame(env_popup, text="Manual Environment Input", padding=10)

        def toggle_mode():
            if self.manual_env.get():
                ctd_frame.pack_forget()
                manual_frame.pack(fill=tk.X, padx=10, pady=5)
            else:
                manual_frame.pack_forget()
                ctd_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Checkbutton(
            top_frame,
            text="Set Environment Manually",
            variable=self.manual_env,
            command=toggle_mode,
        ).pack(side=tk.LEFT)

        self.create_file_entry(
            ctd_frame,
            "CTD File:",
            self.ctd_file,
            False,
            dialog_parent=env_popup,
        )

        calc_frame = ttk.LabelFrame(ctd_frame, text="Calculate Average from CTD", padding=10)
        calc_frame.pack(fill=tk.X, padx=5, pady=10)

        depth_frame = ttk.Frame(calc_frame)
        depth_frame.pack(fill=tk.X, pady=5)
        ttk.Label(depth_frame, text="Transducer Depth (m):").pack(side=tk.LEFT)
        ttk.Entry(depth_frame, textvariable=self.transducer_depth, width=10).pack(side=tk.LEFT, padx=5)

        end_depth_frame = ttk.Frame(calc_frame)
        end_depth_frame.pack(fill=tk.X, pady=5)
        ttk.Label(end_depth_frame, text="End Depth (m):").pack(side=tk.LEFT)
        end_depth_var = tk.DoubleVar(value=50.0)
        ttk.Entry(end_depth_frame, textvariable=end_depth_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            calc_frame,
            text="Calculate",
            command=lambda: self.calculate_ctd_averages(
                end_depth_var.get(), self.transducer_depth.get(), parent=env_popup
            ),
        ).pack(pady=5)

        manual_grid = ttk.Frame(manual_frame)
        manual_grid.pack(fill=tk.X, pady=5)

        ttk.Label(manual_grid, text="Temperature (°C):").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        ttk.Entry(manual_grid, textvariable=self.manual_temp, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(manual_grid, text="Salinity (PSU):").grid(row=1, column=0, sticky='e', padx=5, pady=2)
        ttk.Entry(manual_grid, textvariable=self.manual_sal, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(manual_grid, text="Sound Speed (m/s):").grid(row=2, column=0, sticky='e', padx=5, pady=2)
        ttk.Entry(manual_grid, textvariable=self.manual_c, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=2)

        def save_and_close():
            self.update_env_status_label()
            env_popup.destroy()

        ttk.Button(env_popup, text="Save and Close", command=save_and_close).pack(pady=10)
        toggle_mode()

    def calculate_ctd_averages(self, end_depth, transducer_depth=None, parent=None):
        ctd_path = self.ctd_file.get()
        if not ctd_path or not os.path.exists(ctd_path):
            messagebox.showerror("Error", "Please select a valid CTD file first.", parent=parent)
            return

        try:
            if transducer_depth is None:
                transducer_depth = self.transducer_depth.get()
            if transducer_depth >= end_depth:
                messagebox.showwarning(
                    "Warning",
                    "Transducer depth must be less than the end depth.",
                    parent=parent,
                )
                return

            ctd_df = read_ctd_file(ctd_path)
            mask = (ctd_df['depth'] >= transducer_depth) & (ctd_df['depth'] <= end_depth)
            if not mask.any():
                messagebox.showwarning(
                    "Warning",
                    f"No CTD data found in the specified depth range "
                    f"({transducer_depth}m to {end_depth}m)",
                    parent=parent,
                )
                return

            profile_df = (
                ctd_df[['depth', 'temp', 'sal']]
                .dropna()
                .sort_values('depth')
            )
            avg_df = profile_df[
                (profile_df['depth'] >= transducer_depth)
                & (profile_df['depth'] <= end_depth)
            ]
            avg_temp = avg_df['temp'].mean()
            avg_sal = avg_df['sal'].mean()
            mid_depth = (transducer_depth + end_depth) / 2
            avg_sound_speed, _ = tsCalc.water_properties(
                np.array([avg_sal]),
                np.array([avg_temp]),
                np.array([mid_depth]),
                lon=0.0,
                lat=55.0,
            )

            transducer_temp = np.interp(
                transducer_depth, profile_df['depth'], profile_df['temp']
            )
            transducer_sal = np.interp(
                transducer_depth, profile_df['depth'], profile_df['sal']
            )
            transducer_sound_speed, _ = tsCalc.water_properties(
                np.array([transducer_sal]),
                np.array([transducer_temp]),
                np.array([transducer_depth]),
                lon=0.0,
                lat=55.0,
            )

            messagebox.showinfo(
                "CTD Calculation Results",
                (
                    f"Values at transducer depth ({transducer_depth}m):\n"
                    f"  - Temperature: {transducer_temp:.3f} °C\n"
                    f"  - Salinity: {transducer_sal:.3f} PSU\n"
                    f"  - Sound Speed: {transducer_sound_speed.item():.3f} m/s\n\n"
                    f"Average values from {transducer_depth}m to {end_depth}m:\n"
                    f"  - Temperature: {avg_temp:.3f} °C\n"
                    f"  - Salinity: {avg_sal:.3f} PSU\n"
                    f"  - Sound Speed: {avg_sound_speed.item():.3f} m/s"
                ),
                parent=parent,
            )
        except Exception as exc:
            messagebox.showerror(
                "CTD Calculation Error", f"An error occurred: {exc}", parent=parent
            )

    def update_env_status_label(self):
        if self.manual_env.get():
            self.env_status_label.config(text="Using Manual Values", foreground="blue")
        elif self.ctd_file.get():
            filename = os.path.basename(self.ctd_file.get())
            self.env_status_label.config(text=f"CTD: {filename}", foreground="green")
        else:
            self.env_status_label.config(text="CTD not set", foreground="red")

    def create_file_entry(self, parent, label, var, is_dir=False, dialog_parent=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        label_widget = ttk.Label(frame, text=label, width=15, anchor="e")
        label_widget.pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        def browse():
            if is_dir:
                path = filedialog.askdirectory(parent=dialog_parent)
            else:
                path = filedialog.askopenfilename(parent=dialog_parent)
            if path:
                var.set(path)
                
        button = ttk.Button(frame, text="Browse", command=browse)
        button.pack(side=tk.LEFT)
        return {"frame": frame, "label": label_widget, "entry": entry, "button": button}

    def update_mode_label(self):
        mode_name = "AVO" if self.calibration_mode == CONFIG_MODE_AVO else "QuickCal"
        self.mode_label_var.set(f"Mode: {mode_name}")

    def apply_mode_state(self):
        self.update_mode_label()
        widget_state = "disabled" if self.calibration_mode == CONFIG_MODE_AVO else "normal"
        for widget in self.quickcal_only_widgets:
            widget.configure(state=widget_state)

    # --- Channel Management ---
    
    def refresh_channel_list(self):
        self.channel_listbox.delete(0, tk.END)
        for ch in self.channels:
            files_count = len(ch.get('raw_files', []))
            s_range = ch.get('sphere_range', 'N/A')
            
            # Helper text for display
            extra = ""
            if self.calibration_mode != CONFIG_MODE_AVO:
                if ch.get('min_ts') and ch.get('max_ts'):
                    extra = f" | TS: {ch.get('min_ts')} to {ch.get('max_ts')} dB"
                elif ch.get('sphere_ts_tolerance'):
                    extra = f" | TS Tol: {ch.get('sphere_ts_tolerance')} dB"

            txt = f"{ch.get('id', 'Unknown')} | Range: {s_range}m | Files: {files_count}{extra}"
            self.channel_listbox.insert(tk.END, txt)

    def edit_selected_channel(self):
        sel = self.channel_listbox.curselection()
        if not sel: return
        idx = sel[0]
        self.open_channel_popup(idx)

    def open_channel_popup(self, index=None):
        is_edit = (index is not None)
        title = "Edit Channel" if is_edit else "Add Channel"
        is_avo_mode = self.calibration_mode == CONFIG_MODE_AVO
        
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("500x480" if is_avo_mode else "520x780")
        self._keep_child_window_in_front(popup)
        
        # Initialize Variables
        c_id = tk.StringVar()
        c_range = tk.DoubleVar(value=21.0)
        c_size = tk.DoubleVar(value=0.0) # 0 means use global
        c_mat = tk.StringVar() # Empty means use global
        c_bad_data = tk.StringVar()
        
        # TS Override variables (StringVar allows empty check)
        c_ts_tol = tk.StringVar()
        c_min_ts = tk.StringVar()
        c_max_ts = tk.StringVar()
        c_det_PLDL = tk.StringVar()
        c_det_maxNormPulseLen = tk.StringVar()
        c_det_minNormPulseLen = tk.StringVar()
        c_det_maxBeamComp = tk.StringVar()
        c_det_maxSDalong = tk.StringVar()
        c_det_maxSDathwart = tk.StringVar()
        c_det_min_thresh = tk.StringVar()
        c_det_max_thresh = tk.StringVar()

        files_list = []
        
        # If Editing, populate data
        if is_edit:
            data = self.channels[index]
            c_id.set(data.get('id', ''))
            c_range.set(data.get('sphere_range', 21.0))
            c_size.set(data.get('sphere_size', 0.0))
            c_mat.set(data.get('sphere_material', ''))
            c_bad_data.set(data.get('bad_data_regions_file', ''))
            
            # TS params
            if 'sphere_ts_tolerance' in data:
                c_ts_tol.set(str(data['sphere_ts_tolerance']))
            if 'min_ts' in data:
                c_min_ts.set(str(data['min_ts']))
            if 'max_ts' in data:
                c_max_ts.set(str(data['max_ts']))

            if 'detection_parameters' in data:
                detection_parameters = data['detection_parameters']
                c_det_PLDL.set(detection_parameters.get('PLDL', ''))
                c_det_maxNormPulseLen.set(detection_parameters.get('maxNormPulseLen', ''))
                c_det_minNormPulseLen.set(detection_parameters.get('minNormPulseLen', ''))
                c_det_maxBeamComp.set(detection_parameters.get('maxBeamComp', ''))
                c_det_maxSDalong.set(detection_parameters.get('maxSDalong', ''))
                c_det_maxSDathwart.set(detection_parameters.get('maxSDathwart', ''))
                c_det_min_thresh.set(detection_parameters.get('min_threshold', ''))
                c_det_max_thresh.set(detection_parameters.get('max_threshold', ''))

            files_list = data.get('raw_files', [])[:] # Copy list
        
        # UI Elements
        # Basic info
        info_frame = ttk.LabelFrame(popup, text="Channel Info", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(info_frame, text="Channel ID:").grid(row=0, column=0, sticky="e")
        ttk.Entry(info_frame, textvariable=c_id, width=35).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(info_frame, text="Sphere Range (m):").grid(row=1, column=0, sticky="e")
        ttk.Entry(info_frame, textvariable=c_range, width=15).grid(row=1, column=1, sticky="w", padx=5)

        if not is_avo_mode:
            ttk.Label(info_frame, text="Sphere Size (mm):").grid(row=2, column=0, sticky="e")
            ttk.Entry(info_frame, textvariable=c_size, width=15).grid(row=2, column=1, sticky="w", padx=5)
            ttk.Label(info_frame, text="(0 = Use Global)").grid(row=2, column=2, sticky="w")

            ttk.Label(info_frame, text="Sphere Material:").grid(row=3, column=0, sticky="e")
            ttk.Entry(info_frame, textvariable=c_mat, width=15).grid(row=3, column=1, sticky="w", padx=5)
            ttk.Label(info_frame, text="(Empty = Use Global)").grid(row=3, column=2, sticky="w")

            ttk.Label(info_frame, text="Bad Data File:").grid(row=4, column=0, sticky="e")
            ttk.Entry(info_frame, textvariable=c_bad_data, width=25).grid(row=4, column=1, sticky="w", padx=5)
            def browse_channel_bad_data():
                p = filedialog.askopenfilename(
                    filetypes=[("Echoview Region Files", "*.evr"), ("All files", "*.*")],
                    title="Select Channel Bad Data Regions File",
                    parent=popup,
                )
                if p:
                    c_bad_data.set(p)
            c_bd_btn_frame = ttk.Frame(info_frame)
            c_bd_btn_frame.grid(row=4, column=2, sticky="w")
            ttk.Button(c_bd_btn_frame, text="Browse...", command=browse_channel_bad_data).pack(side=tk.LEFT)
            ttk.Button(c_bd_btn_frame, text="Clear", command=lambda: c_bad_data.set("")).pack(side=tk.LEFT, padx=2)

            ts_frame = ttk.LabelFrame(popup, text="TS Filtering Overrides (Leave empty to use Global defaults)", padding=10)
            ts_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(ts_frame, text="TS Tolerance (+/- dB):").grid(row=0, column=0, sticky="e")
            ttk.Entry(ts_frame, textvariable=c_ts_tol, width=10).grid(row=0, column=1, sticky="w", padx=5)

            ttk.Label(ts_frame, text="OR Explicit Range (Overrides Tolerance):", font='TkDefaultFont 9 bold').grid(row=1, column=0, columnspan=2, sticky="w", pady=(5,2))

            ttk.Label(ts_frame, text="Min TS (dB):").grid(row=2, column=0, sticky="e")
            ttk.Entry(ts_frame, textvariable=c_min_ts, width=10).grid(row=2, column=1, sticky="w", padx=5)

            ttk.Label(ts_frame, text="Max TS (dB):").grid(row=2, column=2, sticky="e")
            ttk.Entry(ts_frame, textvariable=c_max_ts, width=10).grid(row=2, column=3, sticky="w", padx=5)

            det_frame = ttk.LabelFrame(popup, text="Detection Parameter Overrides (Leave empty to use Global)", padding=10)
            det_frame.pack(fill=tk.X, padx=10, pady=5)

            det_params_vars = [
                ("PLDL", c_det_PLDL),
                ("Max Norm Pulse Len", c_det_maxNormPulseLen),
                ("Min Norm Pulse Len", c_det_minNormPulseLen),
                ("Max Beam Comp", c_det_maxBeamComp),
                ("Max SD Along", c_det_maxSDalong),
                ("Max SD Athwart", c_det_maxSDathwart),
                ("Min Threshold", c_det_min_thresh),
                ("Max Threshold", c_det_max_thresh),
            ]

            for i, (label, var) in enumerate(det_params_vars):
                r, c = divmod(i, 2)
                ttk.Label(det_frame, text=label + ":").grid(row=r, column=c * 2, sticky="e", padx=5, pady=2)
                ttk.Entry(det_frame, textvariable=var, width=10).grid(row=r, column=c * 2 + 1, sticky="w", padx=5, pady=2)

        # Files Listbox
        f_frame = ttk.LabelFrame(popup, text="Raw Files")
        f_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        lst = tk.Listbox(f_frame)
        lst.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(f_frame, command=lst.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lst.config(yscrollcommand=sb.set)
        
        # Populate Listbox initially
        for f in files_list:
            lst.insert(tk.END, os.path.basename(f))
        
        def add_files():
            fs = filedialog.askopenfilenames(
                filetypes=[("Raw Files", "*.raw"), ("All Files", "*.*")],
                parent=popup,
            )
            for f in fs:
                if f not in files_list:
                    files_list.append(f)
                    lst.insert(tk.END, os.path.basename(f))
        
        def remove_file():
            sel_f = lst.curselection()
            if not sel_f: return
            idx_f = sel_f[0]
            del files_list[idx_f]
            lst.delete(idx_f)

        def save_ch():
            if not c_id.get():
                messagebox.showerror("Error", "Channel ID is required")
                return
            if not files_list:
                messagebox.showerror("Error", "At least one raw file is required")
                return
            
            ch_data = {
                'id': c_id.get(),
                'raw_files': files_list,
                'sphere_range': c_range.get()
            }
            if not is_avo_mode:
                if c_size.get() > 0:
                    ch_data['sphere_size'] = c_size.get()
                if c_mat.get():
                    ch_data['sphere_material'] = c_mat.get()
                if c_bad_data.get().strip():
                    ch_data['bad_data_regions_file'] = c_bad_data.get().strip()

                try:
                    if c_ts_tol.get().strip():
                        ch_data['sphere_ts_tolerance'] = float(c_ts_tol.get())
                    if c_min_ts.get().strip():
                        ch_data['min_ts'] = float(c_min_ts.get())
                    if c_max_ts.get().strip():
                        ch_data['max_ts'] = float(c_max_ts.get())
                except ValueError:
                    messagebox.showerror("Error", "TS parameters must be numeric")
                    return

                det_data = {}
                det_params_to_save = {
                    'PLDL': c_det_PLDL,
                    'maxNormPulseLen': c_det_maxNormPulseLen,
                    'minNormPulseLen': c_det_minNormPulseLen,
                    'maxBeamComp': c_det_maxBeamComp,
                    'maxSDalong': c_det_maxSDalong,
                    'maxSDathwart': c_det_maxSDathwart,
                    'min_threshold': c_det_min_thresh,
                    'max_threshold': c_det_max_thresh,
                }
                try:
                    for name, var in det_params_to_save.items():
                        if var.get().strip():
                            det_data[name] = float(var.get())
                except ValueError:
                    messagebox.showerror("Error", "Detection parameters must be numeric")
                    return

                if det_data:
                    ch_data['detection_parameters'] = det_data

            if is_edit:
                self.channels[index] = ch_data
            else:
                self.channels.append(ch_data)
                
            self.refresh_channel_list()
            popup.destroy()

        btn_bar = ttk.Frame(popup)
        btn_bar.pack(pady=10)
        ttk.Button(btn_bar, text="Add Files", command=add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_bar, text="Remove File", command=remove_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_bar, text="Save Channel", command=save_ch).pack(side=tk.LEFT, padx=5)

    def remove_channel(self):
        sel = self.channel_listbox.curselection()
        if not sel: return
        idx = sel[0]
        del self.channels[idx]
        self.refresh_channel_list()

    def clear_channels(self):
        self.channels = []
        self.refresh_channel_list()

    def generate_bad_data_regions_file(self):
        """Generate an Echoview file with all raw files for bad data region definition."""
        all_raw_files = []
        for channel in self.channels:
            all_raw_files.extend(channel.get('raw_files', []))

        if not all_raw_files:
            messagebox.showerror("Error", "No raw files found in any channel.")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".ev",
            filetypes=[("Echoview Files", "*.ev")],
            title="Save Echoview File As",
        )
        if not output_path:
            return

        self.bad_data_button.configure(state="disabled")
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Creating EV File")
        progress_dialog.transient(self.root)
        progress_dialog.resizable(False, False)
        ttk.Label(
            progress_dialog,
            text="The EV file is being created. Please wait...",
            padding=20,
        ).pack()
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.update_idletasks()

        try:
            self.generate_ev_file(expand_raw_files(all_raw_files), output_path)
            progress_dialog.destroy()
            messagebox.showinfo(
                "Next Steps",
                (
                    f"Echoview file '{os.path.basename(output_path)}' created.\n\n"
                    "1. Open the generated file in Echoview.\n"
                    "2. On a single channel, identify and mark bad data regions.\n"
                    "3. Export the region definitions (.evr) via Export > Regions > Definitions.\n"
                    "4. Click 'Load bad data regions' and select the exported .evr file."
                ),
            )
        except Exception as exc:
            if progress_dialog.winfo_exists():
                progress_dialog.destroy()
            messagebox.showerror("Echoview Error", f"Failed to generate EV file: {exc}")
        finally:
            self.bad_data_button.configure(state="normal")

    def load_bad_data_regions(self):
        """Load a bad data regions file."""
        path = filedialog.askopenfilename(
            filetypes=[("Echoview Region Files", "*.evr"), ("All files", "*.*")],
            title="Select Bad Data Regions File",
        )
        if path:
            self.bad_data_regions_file.set(path)
            print(f"Loaded bad data regions file: {path}")

    def generate_ev_file(self, raw_files, output_path):
        """Generates an Echoview file using COM."""
        try:
            print("Connecting to Echoview...")
            ev_app = win32com.client.Dispatch("EchoviewCom.EvApplication")
            ev_app.Minimize()

            print("Creating new empty EV file")
            ev_file = ev_app.NewFile()
            for raw_file in raw_files:
                print(raw_file)
                ev_file.Filesets.Item(0).DataFiles.Add(raw_file)

            print(f"Saving new EV file to: {output_path}")
            ev_file.SaveAs(output_path)
            print("EV file generation complete.")
        except Exception:
            raise

    # --- YAML IO ---

    def generate_config_dict(self):
        config = {
            'output_directory': self.output_dir.get(),
            'default_ctd': self.ctd_file.get(),
            'bad_data_regions_file': self.bad_data_regions_file.get(),
            'default_sphere_size': self.sphere_size.get(),
            'default_sphere_material': self.sphere_mat.get(),
            'sphere_range_tolerance': self.range_tol.get(),
            'sphere_ts_tolerance': self.ts_tol.get(),
            'environment_settings': {
                'manual_env': self.manual_env.get(),
                'manual_temp': self.manual_temp.get(),
                'manual_sal': self.manual_sal.get(),
                'manual_c': self.manual_c.get(),
                'transducer_depth': self.transducer_depth.get(),
            },
            'detection_parameters': {
                'PLDL': self.det_PLDL.get(),
                'maxNormPulseLen': self.det_maxNormPulseLen.get(),
                'minNormPulseLen': self.det_minNormPulseLen.get(),
                'maxBeamComp': self.det_maxBeamComp.get(),
                'maxSDalong': self.det_maxSDalong.get(),
                'maxSDathwart': self.det_maxSDathwart.get(),
                'min_threshold': self.det_min_thresh.get(),
                'max_threshold': self.det_max_thresh.get()
            },
            'channels': self.channels
        }
        if self.calibration_mode == CONFIG_MODE_AVO:
            config = {
                'calibration_mode': CONFIG_MODE_AVO,
                'output_directory': self.output_dir.get(),
                'default_ctd': self.ctd_file.get(),
                'bad_data_regions_file': self.bad_data_regions_file.get(),
                'environment_settings': {
                    'transducer_depth': self.transducer_depth.get(),
                },
                'avo_settings': copy.deepcopy(self.avo_settings),
                'channels': self.channels,
            }
        return config

    def load_yaml(self):
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yml *.yaml")])
        if not path: return
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f) or {}

            self.calibration_mode = normalize_calibration_mode(config.get('calibration_mode'))
            self.avo_settings = merge_avo_settings(config.get('avo_settings', {}))
            
            self.output_dir.set(config.get('output_directory', ''))
            self.ctd_file.set(config.get('default_ctd', ''))
            self.bad_data_regions_file.set(config.get('bad_data_regions_file', ''))

            env = config.get('environment_settings', {})
            self.manual_env.set(env.get('manual_env', False))
            self.manual_temp.set(env.get('manual_temp', 10.0))
            self.manual_sal.set(env.get('manual_sal', 35.0))
            self.manual_c.set(env.get('manual_c', 1500.0))
            self.transducer_depth.set(env.get('transducer_depth', 9.15))

            self.sphere_size.set(config.get('default_sphere_size', 38.1))
            self.sphere_mat.set(config.get('default_sphere_material', 'Tungsten carbide'))
            self.range_tol.set(config.get('sphere_range_tolerance', 1.0))
            self.ts_tol.set(config.get('sphere_ts_tolerance', 1.0))
            
            dp = config.get('detection_parameters', {})
            self.det_PLDL.set(dp.get('PLDL', 6))
            self.det_maxNormPulseLen.set(dp.get('maxNormPulseLen', 20))
            self.det_minNormPulseLen.set(dp.get('minNormPulseLen', 0.1))
            self.det_maxBeamComp.set(dp.get('maxBeamComp', 0.1))
            self.det_maxSDalong.set(dp.get('maxSDalong', 0.6))
            self.det_maxSDathwart.set(dp.get('maxSDathwart', 0.6))
            self.det_min_thresh.set(dp.get('min_threshold', -50))
            self.det_max_thresh.set(dp.get('max_threshold', -20))
            
            self.channels = config.get('channels', []) or []
            self.apply_mode_state()
            self.refresh_channel_list()
            self.update_env_status_label()
            print(f"Loaded configuration from {path} ({self.calibration_mode})")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def save_yaml_as(self):
        path = filedialog.asksaveasfilename(defaultextension=".yml", filetypes=[("YAML", "*.yml")])
        if not path: return
        try:
            config = self.generate_config_dict()
            with open(path, 'w') as f:
                f.write(format_saved_config_yaml(config))
            print(f"Saved configuration to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def run_calibration_thread(self):
        if self.calibration_running:
            return

        # Save temp config first
        config = self.generate_config_dict()
        if not config['channels']:
            messagebox.showwarning("Warning", "No channels defined.")
            return

        temp_path = os.path.join(os.getcwd(), "_gui_temp_config_ev_avo.yaml")
        with open(temp_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

        self.calibration_running = True
        self.run_button.configure(state="disabled")
            
        # Run in thread to keep GUI responsive
        t = threading.Thread(target=self.run_logic, args=(temp_path,))
        t.start()

    def run_logic(self, config_path):
        print("--- Starting Calibration ---")
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file) or {}

            mode = normalize_calibration_mode(config.get('calibration_mode'))
            if mode == CONFIG_MODE_AVO:
                run_avo_batch_calibration(config_path, do_plot=True)
            else:
                run_batch_calibration(config_path, do_plot=True)
            print("--- Calibration Finished ---")
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temp file
            if os.path.exists(config_path):
                try:
                    os.remove(config_path)
                except:
                    pass

            self.root.after(0, self._calibration_finished)

    def _calibration_finished(self):
        """Re-enable calibration controls on Tk's main thread."""
        self.calibration_running = False
        self.run_button.configure(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = QuickCalGUI(root)
    root.mainloop()