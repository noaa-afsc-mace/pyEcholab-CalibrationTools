import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ctd
import gsw
import tsCalc
import sys
import yaml
from glob import glob
from scipy.signal import argrelextrema

# Ensure MACEFunctions are accessible
sys.path.insert(0, "G:/WPy64-31150/applications/MaceFunctions")
from echolab2.instruments import echosounder
from echolab2.plotting.matplotlib import echogram
from echolab2.processing import line, grid, integration
from matplotlib.pyplot import figure, show

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

    def run_calibration(self, range_tolerance=1.0, ts_tolerance=1.0, plot=True, plot_save_dir=None):
        print(f"Running calibration for {self.channel_id} with {self.sphere_size} mm sphere" )
        ref_ts = self.get_reference_ts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.detect_targets()        
        
        mask = (np.abs(self.targets.r - self.sphere_range) < range_tolerance) & \
               (np.abs(self.targets.cTS - ref_ts) < ts_tolerance)
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
    for ch_conf in config['channels']:
        channel_id = ch_conf['id']
        ch_detect = global_detect_params.copy()
        ch_detect.update(ch_conf.get('detection_parameters', {}))
        
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
                ts_tolerance=ch_conf.get('sphere_ts_tolerance', global_ts_tol),
                plot=do_plot, plot_save_dir=plots_dir
            )
            if res: all_results.append(res)
        except Exception as e:
            print(f"FAILED {channel_id}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(base_dir, "CalibrationSummary.csv"), index=False)
    print('Complete, see %s for results' % os.path.join(base_dir, "CalibrationSummary.csv"))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_batch_calibration(sys.argv[1], do_plot=True)
    else:
        print("Usage: python QuickCal.py <config.yaml>")