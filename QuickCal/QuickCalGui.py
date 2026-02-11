import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to prevent thread issues with Matplotlib if strictly saving plots,
# or standard backend if interactive. Defaulting to standard but handling closes.
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import ctd
import gsw
import sys
import yaml
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from glob import glob
from scipy.signal import argrelextrema

# Ensure MACEFunctions are accessible
sys.path.insert(0, "G:/WPy64-31150/applications/MaceFunctions")

# Try imports - handle gracefully if running in environment without them for UI testing
try:
    import tsCalc
    from echolab2.instruments import echosounder
    from echolab2.plotting.matplotlib import echogram
    from echolab2.processing import line, grid, integration
    from matplotlib.pyplot import figure, show
except ImportError as e:
    print(f"Warning: Critical dependencies missing ({e}). GUI will load but calibration will fail.")

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
    def __init__(self, channel_id, raw_files, ctd_file, sphere_size=38.1, sphere_mat='Tungsten carbide', sphere_range=21.0, detect_config=None):
        self.channel_id = channel_id
        self.raw_files = self._build_file_list(raw_files)
        self.ctd_file = ctd_file
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

    def _build_file_list(self, files):
        """Expands wildcards and ensures a sorted list of unique files."""
        if not isinstance(files, list):
            files = [files]
        
        expanded_files = []
        for f in files:
            # Handle wildcards (e.g., *.raw)
            matches = glob(f)
            if matches:
                expanded_files.extend(matches)
            else:
                # Fallback if glob returns nothing (e.g., direct path not found)
                expanded_files.append(f)
        
        return sorted(list(set(expanded_files)))

    def load_data(self):
        """Reads raw data and extracts calibration/angle information."""
        print(f"Loading {len(self.raw_files)} files for channel {self.channel_id}...")
        self.ek_data = echosounder.read(self.raw_files, channel_ids=[self.channel_id])
        self.cal = echosounder.get_calibration_from_raw(self.ek_data)[self.channel_id]
        
        chan_obj = self.ek_data.get_channel_data()[self.channel_id][0]
        self.along, self.athwart = chan_obj.get_physical_angles(calibration=self.cal)
        
        gga = chan_obj.nmea_data.get_datagrams('GGA')['GGA']['data'][0]
        self.lat = float(gga.lat[:2]) + (float(gga.lat[2:]) / 60)
        self.lon = float(gga.lon[:3]) + (float(gga.lon[3:]) / 60)
        
        self.d_sv = echosounder.get_Sv(self.ek_data)[self.channel_id]
        self.d_sp = echosounder.get_Sp(self.ek_data)[self.channel_id]

    def get_reference_ts(self):
        tsbw = {18000:{512:1750,1024:1570}, 38000:{512:3280,1024:2430}, 
                70000:{512:4630,1024:2830}, 120000:{512:5490,1024:2990}, 
                200000:{512:590,1024:3050}, 333000:{512:590,1024:3050}}
        
        ext = self.ctd_file.split('.')[-1].lower()
        if ext == 'cnv':
            df = ctd.from_cnv(self.ctd_file)[['t090C','sal00']].rename(columns={"t090C": "temp", "sal00": "sal"})
            df['depth'] = gsw.z_from_p(df.index, 50)
            df = df.reset_index()
        else:
            df = pd.read_csv(self.ctd_file)
            df.rename(columns={'Depth (Meter)':'depth','Temperature (Celsius)':'temp',
                               'Salinity (Practical Salinity Scale)':'sal'}, inplace=True)

        material = tsCalc.material_properties()[self.sphere_mat]
        sphere_env = df.iloc[(df['depth'] - np.floor(self.sphere_range)-9.15).abs().argsort()[:1]]
        
        c, rho = tsCalc.water_properties(sphere_env.sal.values, sphere_env.temp.values, 
                                         sphere_env.depth.values, lon=0.0, lat=self.lat)
        
        f = int(self.cal.frequency)
        pl = int(np.round(self.cal.pulse_duration * 1e6))
        bw = tsbw[f][pl]
        
        fr, ts = tsCalc.freq_response(f-bw/2, f+bw/2, self.sphere_size/1000/2, c, 
                                      material['c1'], material['c2'], rho, material['rho1'], fstep=100)
        return 10 * np.log10(np.mean(10**(ts/10)))

    def detect_targets(self):
        for ping in range(self.d_sp.n_pings):
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
        config = yaml.safe_load(f)
    
    base_dir = config.get('output_directory', './cal_results')
    base_dir = os.path.join(base_dir, 'calibration_output')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    global_ctd = config.get('default_ctd')
    global_sphere = config.get('default_sphere_size', 38.1)
    global_sphere_mat = config.get('default_sphere_material', 38.1)
    global_range_tol = config.get('sphere_range_tolerance', 1.0)
    global_ts_tol = config.get('sphere_ts_tolerance', 1.0)
    global_detect_params = config.get('detection_parameters', {})

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
        df.to_csv(os.path.join(base_dir, "CalibrationSummary.csv"), index=False)
        print('Complete, see %s for results' % os.path.join(base_dir, "CalibrationSummary.csv"))
    else:
        print("No results generated.")

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
        self.root.title("QuickCal - Echosounder Calibration")
        self.root.geometry("900x800")

        # Variables for Global Settings
        self.output_dir = tk.StringVar()
        self.ctd_file = tk.StringVar()
        self.sphere_size = tk.DoubleVar(value=38.1)
        self.sphere_mat = tk.StringVar(value="Tungsten carbide")
        self.range_tol = tk.DoubleVar(value=2.0)
        self.ts_tol = tk.DoubleVar(value=1.0)
        
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

    def create_widgets(self):
        # Top Toolbar (Load/Save)
        top_frame = ttk.Frame(self.root, padding="5")
        top_frame.pack(fill=tk.X)
        ttk.Button(top_frame, text="Load Config (YAML)", command=self.load_yaml).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Save Config (YAML)", command=self.save_yaml_as).pack(side=tk.LEFT, padx=5)
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
        self.create_file_entry(f1, "CTD File:", self.ctd_file, False)
        
        grid_f1 = ttk.Frame(f1)
        grid_f1.pack(fill=tk.X, pady=5)
        ttk.Label(grid_f1, text="Default Sphere Size (mm):").grid(row=0, column=0, sticky="e")
        ttk.Entry(grid_f1, textvariable=self.sphere_size, width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(grid_f1, text="Sphere Material:").grid(row=0, column=2, sticky="e")
        ttk.Entry(grid_f1, textvariable=self.sphere_mat, width=20).grid(row=0, column=3, sticky="w", padx=5)

        ttk.Label(grid_f1, text="Range Tolerance (m):").grid(row=1, column=0, sticky="e")
        ttk.Entry(grid_f1, textvariable=self.range_tol, width=10).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(grid_f1, text="TS Tolerance (dB):").grid(row=1, column=2, sticky="e")
        ttk.Entry(grid_f1, textvariable=self.ts_tol, width=10).grid(row=1, column=3, sticky="w", padx=5)

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
            ttk.Entry(f2, textvariable=var, width=8).grid(row=r, column=c*2+1, sticky="w", padx=5, pady=2)

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

    def create_file_entry(self, parent, label, var, is_dir=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label, width=15, anchor="e").pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        def browse():
            if is_dir:
                path = filedialog.askdirectory()
            else:
                path = filedialog.askopenfilename()
            if path:
                var.set(path)
                
        ttk.Button(frame, text="Browse", command=browse).pack(side=tk.LEFT)

    # --- Channel Management ---
    
    def refresh_channel_list(self):
        self.channel_listbox.delete(0, tk.END)
        for ch in self.channels:
            files_count = len(ch.get('raw_files', []))
            s_range = ch.get('sphere_range', 'N/A')
            
            # Helper text for display
            extra = ""
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
        
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("500x550")
        
        # Initialize Variables
        c_id = tk.StringVar()
        c_range = tk.DoubleVar(value=21.0)
        c_size = tk.DoubleVar(value=0.0) # 0 means use global
        c_mat = tk.StringVar() # Empty means use global
        
        # TS Override variables (StringVar allows empty check)
        c_ts_tol = tk.StringVar()
        c_min_ts = tk.StringVar()
        c_max_ts = tk.StringVar()

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

            files_list = data.get('raw_files', [])[:] # Copy list
        
        # UI Elements
        # Basic info
        info_frame = ttk.LabelFrame(popup, text="Channel Info", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(info_frame, text="Channel ID:").grid(row=0, column=0, sticky="e")
        ttk.Entry(info_frame, textvariable=c_id, width=35).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(info_frame, text="Sphere Range (m):").grid(row=1, column=0, sticky="e")
        ttk.Entry(info_frame, textvariable=c_range, width=15).grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Label(info_frame, text="Sphere Size (mm):").grid(row=2, column=0, sticky="e")
        ttk.Entry(info_frame, textvariable=c_size, width=15).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(info_frame, text="(0 = Use Global)").grid(row=2, column=2, sticky="w")

        ttk.Label(info_frame, text="Sphere Material:").grid(row=3, column=0, sticky="e")
        ttk.Entry(info_frame, textvariable=c_mat, width=15).grid(row=3, column=1, sticky="w", padx=5)
        ttk.Label(info_frame, text="(Empty = Use Global)").grid(row=3, column=2, sticky="w")

        # TS Filtering
        ts_frame = ttk.LabelFrame(popup, text="TS Filtering Overrides (Leave empty to use Global defaults)", padding=10)
        ts_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ts_frame, text="TS Tolerance (+/- dB):").grid(row=0, column=0, sticky="e")
        ttk.Entry(ts_frame, textvariable=c_ts_tol, width=10).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(ts_frame, text="OR Explicit Range (Overrides Tolerance):", font='TkDefaultFont 9 bold').grid(row=1, column=0, columnspan=2, sticky="w", pady=(5,2))
        
        ttk.Label(ts_frame, text="Min TS (dB):").grid(row=2, column=0, sticky="e")
        ttk.Entry(ts_frame, textvariable=c_min_ts, width=10).grid(row=2, column=1, sticky="w", padx=5)
        
        ttk.Label(ts_frame, text="Max TS (dB):").grid(row=2, column=2, sticky="e")
        ttk.Entry(ts_frame, textvariable=c_max_ts, width=10).grid(row=2, column=3, sticky="w", padx=5)

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
            if c_size.get() > 0:
                ch_data['sphere_size'] = c_size.get()
            if c_mat.get():
                ch_data['sphere_material'] = c_mat.get()
            
            # Save TS parameters if present
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
        return config

    def load_yaml(self):
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yml *.yaml")])
        if not path: return
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.output_dir.set(config.get('output_directory', ''))
            self.ctd_file.set(config.get('default_ctd', ''))
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
            
            self.channels = config.get('channels', [])
            self.refresh_channel_list()
            print(f"Loaded configuration from {path}")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def save_yaml_as(self):
        path = filedialog.asksaveasfilename(defaultextension=".yml", filetypes=[("YAML", "*.yml")])
        if not path: return
        try:
            config = self.generate_config_dict()
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Saved configuration to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def run_calibration_thread(self):
        # Save temp config first
        config = self.generate_config_dict()
        if not config['channels']:
            messagebox.showwarning("Warning", "No channels defined.")
            return

        temp_path = os.path.join(os.getcwd(), "_gui_temp_config.yaml")
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)
            
        # Run in thread to keep GUI responsive
        t = threading.Thread(target=self.run_logic, args=(temp_path,))
        t.start()

    def run_logic(self, config_path):
        print("--- Starting Calibration ---")
        try:
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