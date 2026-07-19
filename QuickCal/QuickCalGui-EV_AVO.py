import copy
import io
import json
import os
import re
import warnings
from datetime import datetime

import gsw
import matplotlib
# Use Agg backend to prevent thread issues with Matplotlib if strictly saving plots,
# or standard backend if interactive. Defaulting to standard but handling closes.
matplotlib.use('TkAgg') 
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


def _format_block_mapping(mapping, indent=2):
    lines = []
    pad = " " * indent
    for key, value in mapping.items():
        lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
    return lines


def _format_channel_block(channel, include_avo_settings):
    if include_avo_settings:
        lines = [f"  - id: {_yaml_scalar(channel.get('id', ''))}", "    raw_files:"]
        for raw_file in channel.get("raw_files", []) or []:
            lines.append(f"      - {_yaml_scalar(raw_file)}")
        lines.append(f"    sphere_range: {_yaml_scalar(channel.get('sphere_range', ''))}")
        return lines

    lines = ["    # Channel ID", f"  - id: {_yaml_scalar(channel.get('id', ''))}"]
    lines.append(
        "    # Raw file(s). Use wildcards [.../*.raw'] to capture all raw files in the folder."
    )
    lines.append(f"    raw_files: {_format_inline_list(channel.get('raw_files', []), quoted=True)}")
    lines.append("    # Sphere range (m) during on-axis collection")
    lines.append(f"    sphere_range: {_yaml_scalar(channel.get('sphere_range', ''))}")

    if channel.get("sphere_size") not in (None, ""):
        lines.append("    # Sphere size (mm)")
        lines.append(f"    sphere_size: {_yaml_scalar(channel.get('sphere_size'))}")
    if channel.get("sphere_material"):
        lines.append("    # Sphere material")
        lines.append(f"    sphere_material: {_yaml_scalar(channel.get('sphere_material'))}")
    if channel.get("sphere_ts_tolerance") not in (None, ""):
        lines.append("    # +/- TS around the calculated reference TS")
        lines.append(f"    sphere_ts_tolerance: {_yaml_scalar(channel.get('sphere_ts_tolerance'))}")
    if channel.get("min_ts") not in (None, ""):
        lines.append("    # Explicit TS range override")
        lines.append(f"    min_ts: {_yaml_scalar(channel.get('min_ts'))}")
    if channel.get("max_ts") not in (None, ""):
        if channel.get("min_ts") in (None, ""):
            lines.append("    # Explicit TS range override")
        lines.append(f"    max_ts: {_yaml_scalar(channel.get('max_ts'))}")
    det_params = channel.get("detection_parameters") or {}
    if det_params:
        lines.append("    # Individual detection parameter override example:")
        lines.append("    detection_parameters:")
        lines.extend(_format_block_mapping(det_params, indent=6))
    return lines


def format_saved_config_yaml(config):
    if normalize_calibration_mode(config.get("calibration_mode")) == CONFIG_MODE_AVO:
        lines = [
            "##### Configuration file for QuickCalGui_AVO in AVO mode #####",
            "",
            '# Top-level mode flag: when set to "avo", the GUI runs the integrated',
            "# AVO coverage-check workflow instead of the full QuickCal calibration path.",
            'calibration_mode: "avo"',
            "",
            "### Output definitions ###",
            "",
            "# AVO figures and summary CSV will be written under:",
            "#   <output_directory>/avo_output/",
            "# This replaces figure_folder from AVO.ini.",
            f'output_directory: {_yaml_scalar(config.get("output_directory", ""))}',
            "",
            "### Hidden AVO settings migrated from AVO.ini ###",
            "",
            "avo_settings:",
        ]
        lines.extend(_format_block_mapping(merge_avo_settings(config.get("avo_settings", {})), indent=2))
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
        "### Global definitions ###",
        "",
        "# Output directory for calibration results",
        f'output_directory: {_yaml_scalar(config.get("output_directory", ""))}',
        "",
        "# Full path filename of CTD data, SBE cnv or CastAway csv",
        f'default_ctd: {_yaml_scalar(config.get("default_ctd", ""))}',
        "",
        "# Global sphere parameters (NOTE: channel-specific definitions override these)",
        f'default_sphere_size: {_yaml_scalar(config.get("default_sphere_size", 38.1))}',
        f'default_sphere_material: {_yaml_scalar(config.get("default_sphere_material", "Tungsten carbide"))}',
        "# +/- range around the sphere as defined per channel",
        f'sphere_range_tolerance: {_yaml_scalar(config.get("sphere_range_tolerance", 1))}',
        "# +/- TS around the calculated reference TS",
        f'sphere_ts_tolerance: {_yaml_scalar(config.get("sphere_ts_tolerance", 1))}',
        "",
        "# Global single target detection parameters  (NOTE: channel-specific definitions override these)",
        "detection_parameters:",
    ]
    lines.extend(_format_block_mapping(config.get("detection_parameters", {}), indent=2))
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
        tsbw = {14000:{512:1750,1024:1570}, 18000:{512:1750,1024:1570}, 22000:{512:1750,1024:1570},
                35000:{512:3280,1024:2430}, 38000:{512:3280,1024:2430}, 44000:{512:3280,1024:2430}, 
                57000:{512:4630,1024:2830}, 70000:{512:4630,1024:2830}, 82000:{512:4630,1024:2830},
                98000:{512:5490,1024:2990}, 120000:{512:5490,1024:2990}, 148000:{512:5490,1024:2990}, 
                169000:{512:590,1024:3050}, 200000:{512:590,1024:3050}, 230000:{512:590,1024:3050},
                395000:{512:590,1024:3050}, 338000:{512:590,1024:3050}, 440000:{512:590,1024:3050}}
        
        if self.env_settings.get('manual_env', False):
            print("  > Using manual environment values.")
            temp = self.env_settings['manual_temp']
            sal = self.env_settings['manual_sal']
            sound_speed = self.env_settings['manual_c']
            _, rho = tsCalc.water_properties(
                np.array([sal]),
                np.array([temp]),
                np.array([self.sphere_range]),
                lon=0.0,
                lat=self.lat,
            )
        else:
            print("  > Using CTD file for environment values.")
            if not self.ctd_file or not os.path.exists(self.ctd_file):
                raise ValueError("CTD file not found or specified, but manual environment mode is off.")

            df = read_ctd_file(self.ctd_file)
            df = df.reset_index(drop=True)

            sphere_depth = self.sphere_range + 9.15
            path_df = df[(df['depth'] >= 9) & (df['depth'] <= sphere_depth)].copy()

            if path_df.empty:
                raise ValueError(f"No CTD data found in the depth range 9m to {sphere_depth:.2f}m.")

            _, path_rho = tsCalc.water_properties(
                path_df['sal'].values,
                path_df['temp'].values,
                path_df['pressure'].values,
                lon=0.0,
                lat=self.lat,
            )
            rho = path_rho.mean()

            if 'svel' in path_df.columns:
                print("  > Using sound speed from CTD file.")
                sound_speed = 1 / np.mean(1 / path_df['svel'])
            else:
                print("  > Calculating sound speed from CTD temp/sal/pressure.")
                path_c, _ = tsCalc.water_properties(
                    path_df['sal'].values,
                    path_df['temp'].values,
                    path_df['pressure'].values,
                    lon=0.0,
                    lat=self.lat,
                )
                sound_speed = 1 / np.mean(1 / path_c)

        material = tsCalc.material_properties()[self.sphere_mat]

        cal_sound_speed = np.mean(self.cal.sound_speed)
        if round(sound_speed, 1) != round(cal_sound_speed, 1):
            print(
                f"  > WARNING: Sound speed mismatch. CTD/manual value is {sound_speed:.1f} m/s, "
                f"while calibration object value is {cal_sound_speed:.1f} m/s."
            )
        
        f = int(self.cal.frequency)
        pl = int(np.round(self.cal.pulse_duration * 1e6))
        bw = tsbw[f][pl]
        
        fr, ts = tsCalc.freq_response(f-bw/2, f+bw/2, self.sphere_size/1000/2, sound_speed, 
                                      material['c1'], material['c2'], rho, material['rho1'], fstep=100)
        return 10 * np.log10(np.mean(10**(ts/10)))

    def detect_targets(self):
        ping_times_dt = pd.to_datetime(self.d_sp.ping_time)

        for ping in range(self.d_sp.n_pings):
            ping_time = ping_times_dt[ping]
            if any(start_time <= ping_time <= end_time for start_time, end_time in self.bad_data_regions):
                continue

            abs_coeff = self.cal.absorption_coefficient[ping]
            cpv = 40 * np.log10(self.d_sp.range) + 2 * abs_coeff * self.d_sp.range
            calPower = self.d_sp.data[ping] - cpv
            maxima = argrelextrema(calPower, np.greater)[0]

            for l in maxima:
                pldl_val = calPower[l] - self.params.PLDL
                res = self._calculate_pulse_width(calPower, l, pldl_val)
                if res is None: continue
                normWidth, left, right = res

                if (normWidth > self.params.maxNormPulseLen) or (normWidth < self.params.minNormPulseLen):
                    continue

                peak_along = self.along.data[ping][l]
                peak_athwart = self.athwart.data[ping][l]
                beam_comp = self._get_beam_comp(ping, peak_along, peak_athwart)

                if beam_comp > self.params.maxBeamComp:
                    continue

                if not self._check_stdev(ping, left + 1, right - 1):
                    continue

                r_val = (sum(self.d_sp.range[left+1:right-1] * calPower[left+1:right-1])) / \
                        sum(calPower[left+1:right-1]) - (self.cal.sound_speed * self.cal.pulse_duration) / 4
                
                uTS = calPower[l] + (40 * np.log10(r_val)) + (2 * abs_coeff * r_val)
                cTS = uTS + beam_comp

                if self.params.min_threshold <= cTS <= self.params.max_threshold:
                    self._append_target(ping, r_val, uTS, cTS, peak_athwart, peak_along, normWidth)

    def _calculate_pulse_width(self, power, l, limit):
        r_idx = np.where(power[l:] < limit)[0]
        l_idx = np.where(power[:l] < limit)[0]
        if r_idx.size == 0 or l_idx.size == 0: return None
        
        right, left = l + r_idx[0] - 1, l_idx[-1] + 1
        xLeft = left + (limit - power[left]) / (power[left+1] - power[left])
        xRight = right + (limit - power[right]) / (power[right+1] - power[right])
        return (xRight - xLeft) / 4, left, right

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
            self.detect_targets()        
        
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

        observed_ts = 10 * np.log10(np.mean(10**(self.sphere_targets.cTS / 10)))
        observed_ts_std = np.std(self.sphere_targets.cTS)
        mean_range = np.mean(self.sphere_targets.r)

        if plot:
            clean_id = self.channel_id.replace(' ', '_').replace('-', '_').replace(':', '_')+'-'+str(self.sphere_size).split('.')[0]
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].hist(self.sphere_targets.cTS, bins=100)
            ax[0].set_title(f"TS Distribution: {self.channel_id}")
            ax[1].plot(self.sphere_targets.peakAlong, self.sphere_targets.peakAthwart, '.')
            ax[1].set_title(f"Beam Positions: {self.channel_id}")
            if plot_save_dir:
                plt.savefig(os.path.join(plot_save_dir, f"{clean_id}_stats.png"))
            plt.close(fig)

            fig_echo = figure(figsize=(12, 4))
            eg = echogram.Echogram(fig_echo, d_sv_on_axis, threshold=[-90, -30])
            eg.add_colorbar(fig_echo)
            plt.title(f"Echogram: {self.channel_id}")
            if plot_save_dir:
                plt.savefig(os.path.join(plot_save_dir, f"{clean_id}_echogram.png"))
            plt.close(fig_echo)

        upper_line = line.line(ping_time=d_sv_on_axis.ping_time, data=mean_range - (range_tolerance)) 
        lower_line = line.line(ping_time=d_sv_on_axis.ping_time, data=mean_range + (range_tolerance)) 

        integrator = integration.integrator(min_threshold_applied=False)
        grid_obj = grid.grid(interval_length=10000, interval_axis='ping_number', 
                             data=d_sv_on_axis, layer_axis='range', layer_thickness=100)
        
        integrated = integrator.integrate(d_sv_on_axis, grid_obj, 
                                          exclude_above_line=upper_line, 
                                          exclude_below_line=lower_line)

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
            'new_sv_gain': new_sv_gain,
            'calc_ts_gain': calc_gain,
            'sa_correction': sa_corr,
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
                    bad_data_regions_file=bad_data_regions_file,
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
    def __init__(self, channel_id, raw_files, sphere_range, settings, output_dir):
        self.channel_id = channel_id
        self.raw_files = expand_raw_files(raw_files)
        self.sphere_range = float(sphere_range)
        self.settings = merge_avo_settings(settings)
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
        tsbw = {
            18000: {512: 1750, 1024: 1570},
            38000: {512: 3280, 1024: 2430},
            70000: {512: 4630, 1024: 2830},
            120000: {512: 5490, 1024: 2990},
            200000: {512: 590, 1024: 3050},
            333000: {512: 590, 1024: 3050},
            338000: {512: 590, 1024: 3050},
        }

        if hasattr(self.channel_data, "is_cw") and not self.channel_data.is_cw():
            raise ValueError("AVO mode only supports CW data.")

        if self.frequency not in tsbw or self.pulse_length_us not in tsbw[self.frequency]:
            raise ValueError(
                f"Unsupported frequency/pulse combination for AVO mode: {self.frequency} Hz, {self.pulse_length_us} us."
            )

        material = tsCalc.material_properties()[self.settings["sphere_material"]]
        sound_speed, density = tsCalc.water_properties(
            float(self.settings["salinity"]),
            float(self.settings["temp"]),
            self.sphere_depth,
            lon=0.0,
            lat=self.lat,
        )
        bandwidth = tsbw[self.frequency][self.pulse_length_us]
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
        for ping in range(self.d_sp.n_pings):
            cpv = 40 * np.log10(self.d_sp.range) + 2 * self.cal.absorption_coefficient[ping] * self.d_sp.range
            cal_power = self.d_sp.data[ping] - cpv
            maxima = argrelextrema(cal_power, np.greater)[0]
            pulse_term = (self._first_scalar(self.cal.sound_speed) * self._first_scalar(self.cal.pulse_duration)) / 4

            for peak_index in maxima:
                pldl_val = cal_power[peak_index] - self.params.PLDL
                if np.where(cal_power[peak_index:] < pldl_val)[0].size > 0:
                    right = peak_index + np.where(cal_power[peak_index:] < pldl_val)[0][0] - 1
                else:
                    continue

                if np.where(cal_power[:peak_index] < pldl_val)[0].size > 0:
                    left = np.where(cal_power[:peak_index] < pldl_val)[0][-1] + 1
                else:
                    continue

                x_left = left + (pldl_val - cal_power[left]) / (cal_power[left + 1] - cal_power[left])
                x_right = right + (pldl_val - cal_power[right]) / (cal_power[right + 1] - cal_power[right])
                norm_width = (x_right - x_left) / 4
                if norm_width > self.params.maxNormPulseLen or norm_width < self.params.minNormPulseLen:
                    continue

                start_index = left + 1
                end_index = right - 1
                if (end_index - start_index) < 1:
                    continue

                peak_along_deg = self.along.data[ping][peak_index]
                peak_athwart_deg = self.athwart.data[ping][peak_index]
                along_norm = 2 * peak_along_deg / self.cal.beam_width_alongship[ping]
                athwart_norm = 2 * peak_athwart_deg / self.cal.beam_width_athwartship[ping]
                beam_comp = 6.0206 * (
                    along_norm ** 2 + athwart_norm ** 2 - (0.18 * along_norm ** 2 * athwart_norm ** 2)
                )

                if beam_comp > self.params.maxBeamComp:
                    continue

                along_target = self.along.data[ping][start_index:end_index]
                athwart_target = self.athwart.data[ping][start_index:end_index]
                sd_along = np.std(along_target)
                sd_athwart = np.std(athwart_target)
                if sd_along > self.params.maxSDalong or sd_athwart > self.params.maxSDathwart:
                    continue

                r_val = (
                    sum(self.d_sp.range[start_index:end_index] * cal_power[start_index:end_index])
                    / sum(cal_power[start_index:end_index])
                    - pulse_term
                )
                if r_val > self.params.excludeBelow or r_val < self.params.excludeAbove:
                    continue

                u_ts = cal_power[peak_index] + (40 * np.log10(r_val)) + (2 * self.cal.absorption_coefficient[ping] * r_val)
                c_ts = u_ts + beam_comp
                if c_ts < self.params.threshold_min or c_ts > self.params.threshold_max:
                    continue

                self.targets.ping = np.append(self.targets.ping, ping)
                self.targets.r = np.append(self.targets.r, r_val)
                self.targets.uTS = np.append(self.targets.uTS, u_ts)
                self.targets.cTS = np.append(self.targets.cTS, c_ts)
                self.targets.peakAthwart = np.append(self.targets.peakAthwart, peak_athwart_deg)
                self.targets.peakAlong = np.append(self.targets.peakAlong, peak_along_deg)
                self.targets.sdAlng = np.append(self.targets.sdAlng, sd_along)
                self.targets.sdAthw = np.append(self.targets.sdAthw, sd_athwart)
                self.targets.normWidth = np.append(self.targets.normWidth, norm_width)

    def save_echograms(self):
        calview_depth = {
            18000: 2000,
            38000: 2000,
            70000: 2700,
            120000: 5000,
            200000: 8000,
            333000: 8000,
            338000: 8000,
        }
        clean_id = self._safe_filename()
        vessel = self.settings["vessel"]

        full_indices = np.where(self.d_sv.depth <= calview_depth.get(self.frequency, self.d_sv.depth[-1]))[0]
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
        self.root.title("QuickCal EV / AVO - Echosounder Calibration")
        self.root.geometry("900x800")
        self.calibration_mode = CONFIG_MODE_QUICKCAL
        self.avo_settings = get_default_avo_settings()
        self.mode_label_var = tk.StringVar(value="Mode: QuickCal")
        self.quickcal_only_widgets = []

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
        bad_data_button = ttk.Button(
            top_frame,
            text="Generate Bad Data Regions",
            command=self.generate_bad_data_regions_file,
        )
        bad_data_button.pack(side=tk.RIGHT, padx=5)
        self.quickcal_only_widgets.append(bad_data_button)
        ttk.Button(top_frame, text="RUN CALIBRATION", command=self.run_calibration_thread).pack(side=tk.RIGHT, padx=5)

        # Main Scrollable Area
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

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
        self.quickcal_only_widgets.extend([bad_data_entry, bad_data_load_button])
        
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
        f4 = ttk.LabelFrame(self.root, text="Console Output", padding="5")
        f4.pack(side="bottom", fill="both", expand=True, padx=10, pady=5)
        
        self.console = scrolledtext.ScrolledText(f4, height=10, state="disabled")
        self.console.pack(fill="both", expand=True)

        # Redirect stdout/stderr
        sys.stdout = TextRedirector(self.console, "stdout")
        sys.stderr = TextRedirector(self.console, "stderr")

    def open_env_dialog(self):
        env_popup = tk.Toplevel(self.root)
        env_popup.title("Calculate and Set Environment")
        env_popup.geometry("500x350")

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

        self.create_file_entry(ctd_frame, "CTD File:", self.ctd_file, False)

        calc_frame = ttk.LabelFrame(ctd_frame, text="Calculate Average from CTD", padding=10)
        calc_frame.pack(fill=tk.X, padx=5, pady=10)

        depth_frame = ttk.Frame(calc_frame)
        depth_frame.pack(fill=tk.X, pady=5)
        ttk.Label(depth_frame, text="End Depth (m):").pack(side=tk.LEFT)
        end_depth_var = tk.DoubleVar(value=50.0)
        ttk.Entry(depth_frame, textvariable=end_depth_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            calc_frame,
            text="Calculate",
            command=lambda: self.calculate_ctd_averages(end_depth_var.get()),
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

    def calculate_ctd_averages(self, end_depth):
        ctd_path = self.ctd_file.get()
        if not ctd_path or not os.path.exists(ctd_path):
            messagebox.showerror("Error", "Please select a valid CTD file first.")
            return

        try:
            ctd_df = read_ctd_file(ctd_path)
            mask = (ctd_df['depth'] >= 9.15) & (ctd_df['depth'] <= end_depth)
            if not mask.any():
                messagebox.showwarning(
                    "Warning",
                    f"No CTD data found in the specified depth range (9.15m to {end_depth}m)",
                )
                return

            avg_df = ctd_df[mask]
            avg_temp = avg_df['temp'].mean()
            avg_sal = avg_df['sal'].mean()
            mid_depth = (9.15 + end_depth) / 2
            sound_speed, _ = tsCalc.water_properties(
                np.array([avg_sal]),
                np.array([avg_temp]),
                np.array([mid_depth]),
                lon=0.0,
                lat=55.0,
            )

            messagebox.showinfo(
                "CTD Calculation Results",
                (
                    f"Average values from 9.15m to {end_depth}m:\n"
                    f"  - Temperature: {avg_temp:.3f} °C\n"
                    f"  - Salinity: {avg_sal:.3f} PSU\n"
                    f"  - Sound Speed: {sound_speed.item():.3f} m/s"
                ),
            )
        except Exception as exc:
            messagebox.showerror("CTD Calculation Error", f"An error occurred: {exc}")

    def update_env_status_label(self):
        if self.manual_env.get():
            self.env_status_label.config(text="Using Manual Values", foreground="blue")
        elif self.ctd_file.get():
            filename = os.path.basename(self.ctd_file.get())
            self.env_status_label.config(text=f"CTD: {filename}", foreground="green")
        else:
            self.env_status_label.config(text="CTD not set", foreground="red")

    def create_file_entry(self, parent, label, var, is_dir=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        label_widget = ttk.Label(frame, text=label, width=15, anchor="e")
        label_widget.pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        def browse():
            if is_dir:
                path = filedialog.askdirectory()
            else:
                path = filedialog.askopenfilename()
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
        popup.geometry("500x480" if is_avo_mode else "500x750")
        
        # Initialize Variables
        c_id = tk.StringVar()
        c_range = tk.DoubleVar(value=21.0)
        c_size = tk.DoubleVar(value=0.0) # 0 means use global
        c_mat = tk.StringVar() # Empty means use global
        
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
            fs = filedialog.askopenfilenames(filetypes=[("Raw Files", "*.raw"), ("All Files", "*.*")])
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

        try:
            self.generate_ev_file(expand_raw_files(all_raw_files), output_path)
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
            messagebox.showerror("Echoview Error", f"Failed to generate EV file: {exc}")

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
            'default_sphere_size': self.sphere_size.get(),
            'default_sphere_material': self.sphere_mat.get(),
            'sphere_range_tolerance': self.range_tol.get(),
            'sphere_ts_tolerance': self.ts_tol.get(),
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
        # Save temp config first
        config = self.generate_config_dict()
        if not config['channels']:
            messagebox.showwarning("Warning", "No channels defined.")
            return

        temp_path = os.path.join(os.getcwd(), "_gui_temp_config_ev_avo.yaml")
        with open(temp_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
            
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

if __name__ == "__main__":
    root = tk.Tk()
    app = QuickCalGUI(root)
    root.mainloop()
