import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
import tsCalc
import numpy as np

# For 32-bit compatibility only, can be skipped for 64-bit Python
#import matplotlib
#matplotlib.use('TKAgg')


import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import sys
import configparser
from echolab2.instruments import echosounder
import warnings
warnings.filterwarnings("ignore")

from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap
from echolab2.plotting.matplotlib import echogram
from echolab2.processing import  line, grid, integration
from scipy.signal import argrelextrema
#from triwave_correct import TriwaveCorrect
import matplotlib.patches as patches



import numpy as np
from scipy import optimize
import logging

class TriwaveCorrect():
    '''
    Class containing methods for correcting triangle wave noise in data
    '''
    
    def __init__(self, start_sample, end_sample):
        '''
        Initialize triwave correction parameters
        
        :param start_sample: first sample (index) to use in computing mean ringdown
        :type start_sample: int
        
        :param end_sample: last sample (index) to use in computing mean ringdown
        :type end_sample: int
        '''
        
        self.start_sample = start_sample
        self.end_sample = end_sample


    def triwave_correct(self, data_in):
        '''
        Perform correction on raw power data array
        - Compute linear mean of power between start and end samples
        - Fill NaNs with closest earlier neighbor ping
        - Remove spikes in the triwave raw data, using a fixed 0.05 threshold deviation value
        - Fit a triangle
        - Generate an offset array that is the inverse of the fit triangle centered around 0
        - Correct raw power data
        
        :param data: raw data object, which must contain raw power
        :type data: raw_data object derived from pyecholab2 raw_read method
        
        :returns data:  raw data object with corrected raw power
        :returns fit_results: dictionary of parameters of the fit of triangle wave to raw data
        :returns val: bool for success of fit
        '''
        
        L = data_in.n_samples
        try:
            n = np.log10(np.mean(10**(data_in.power[:,self.start_sample:self.end_sample]), axis=1))
        except NameError:
            logging.error("No raw power defined in raw data object.")
            return data_in, False, False
            
        # fill nans with closest earlier ping
        nan_inds = np.argwhere(np.isnan(n))
        while np.any(nan_inds):
            n[nan_inds] = n[nan_inds-1]
            nan_inds = np.argwhere(np.isnan(n))
        
        # fill infs
        inf_inds = np.argwhere(np.isinf(n))
        if not len(n)==len(inf_inds):
            while np.any(inf_inds):
                n[inf_inds] = n[inf_inds-1]
                inf_inds = np.argwhere(np.isinf(n))
        
        # stomp down spikes- 0.05 dB is a good threshold for amplitude of 1 dB in power
        #bad_inds = np.argwhere(np.diff(n)<-0.05)+1
        #while np.any(bad_inds):
        #    n[bad_inds] = n[bad_inds-1]
        #    bad_inds = np.argwhere(np.diff(n)<-0.05)+1
        
        # Fit triangle to ringdown array
        fit_results= self.fit_triangle(n)
        if fit_results['r_squared'] < 0.9:
            logging.warning("Bad triangle fit with r^2 of {}".format(fit_results['r_squared']))
        else:
            logging.info("Triangle fit with r^2 of {}".format(fit_results['r_squared']))
        
        # Find correction triangle
        generated_triangle_offset = self.general_triangle(np.arange(data_in.shape[0]), A=fit_results['amplitude'],
                    M=2721.0, k = fit_results['period_offset'], C=0, dtype='float32')
        triangle_matrix_correct = np.array([generated_triangle_offset,]*L).transpose()
        
        # Correct raw power in data object
        data_in.power = data_in.power - triangle_matrix_correct
        logging.info("Successfully corrected triangle wave noise in raw power data")
        
        return data_in, fit_results, True
        
        
    def fit_triangle(self, mean_ringdown_vec, amplitude=None, period_offset=None,
                          amplitude_offset=None):
        '''
        :param mean_ringdown_vec:  Array of ping ringdown values.
        
        :returns: k, C, R^2
        
        Attempts to fit the values in mean_ringdown_vec to the triangle
        wave offset.  This function returns the two values, k and C.  k is
        the sample offset from period origin (where the first ping lies along
        the triangle period), C is the mean offset of the triangle wave.  R
        is the "R^2" value of the fit (Coefficient of determination)
        
        '''
        N = len(mean_ringdown_vec)
        n = np.arange(N)
        
        fit_func = lambda p: self.general_triangle(n, p[0], 2721.0, p[1], p[2])
        err_func = lambda p: (mean_ringdown_vec-fit_func(p))
        
        if period_offset is None:
            period_offset = 1360 - np.argmax(mean_ringdown_vec)
            
        if amplitude is None:
            amplitude = 1.0
            
        if amplitude_offset is None:
            amplitude_offset = np.mean(mean_ringdown_vec)
            
        guess = [amplitude, period_offset, amplitude_offset]
    
        fit_results = optimize.leastsq(err_func, guess[:], full_output=True)
        
        fit_params, fit_cov, fit_info, fit_msg, fit_success = fit_results
        
        SStot = sum((mean_ringdown_vec - mean_ringdown_vec.mean())**2)
        
        SSerr = sum(err_func(fit_params)**2)
    
        fit_r_squared = 1 - SSerr/SStot
        
        fit_amplitude, fit_period_offset, fit_amplitude_offset = fit_params
        
        #Negative amplitude -> half-period offset.
        if fit_amplitude < 0:
            fit_amplitude = -fit_amplitude
            fit_period_offset += 2721.0 / 2
        fit_period_offset = fit_period_offset % 2721
        
        if abs(fit_period_offset - 2721) < abs(fit_period_offset):
            fit_period_offset -= 2721

        return dict(period_offset=fit_period_offset,
                    amplitude_offset=fit_amplitude_offset,
                    amplitude=fit_amplitude,
                    r_squared=fit_r_squared)
    
    
    def general_triangle(self, n, A=0.5, M=2721,  k=0, C=0, dtype=None):
        '''
        Finds a general triangle-wave function centered at 0 
        
        :param n: sample index
        :type n: array(int)
        
        :param A: Triangle wave amplitude (1/2 peak-to-peak)
        :type A: float
        
        :param M: Triangle wave period in samples
        :type M: int
        
        :param k: Sample offset
        :type k: int
        
        :param C: Amplitude offset
        :type C: float
        
        :returns triangle: array(int) with the same length as n
        
        '''
        n_div_M = ((n + k) % M) / float(M)
        triangle =  A*(2*abs(2 * (n_div_M - np.floor(n_div_M + 0.5))) - 1) + C
        
        if dtype is not None:
            return triangle.astype(dtype)
        else:
            return triangle
        


class detectParmsInit():
    # Set something up to mimic TS range from lobes
    PLDL = 6
    maxNormPulseLen = 20
    minNormPulseLen = .1
    maxBeamComp = 6
    maxSDalong = 2
    maxSDathwart = 2
    excludeBelow = 1e10
    excludeAbove = 0
    threshold_min = -55
    threshold_max = -30

class singleTargetsInit():
    ping = np.array([])
    r = np.array([])
    uTS = np.array([])
    cTS = np.array([])
    peakAthwart = np.array([])
    peakAlong = np.array([])
    sdAlng = np.array([])
    sdAthw = np.array([])
    normWidth = np.array([])
    
def get_refTS(d, sphere_material,sphere_depth,sphere_diameter,defaults):
    try:
        lat = float(d.nmea_data.get_datagrams('GGA')['GGA']['data'][0].lat[:2])+\
        (float(d.nmea_data.get_datagrams('GGA')['GGA']['data'][0].lat[2:])/60)
    except:
        print('No GPS data was found in the raw file.\n')
        if defaults['prompt_for_params']=='True':
            lat = input('Enter the approximate latitude of the calibration in decimal degrees: ')
        else:
            lat = defaults['lat']
        lat = float(lat)

    materialPoperties = tsCalc.material_properties()[sphere_material]

    if defaults['prompt_for_params']=='True':
        temp = input('Enter the temperature of the calibration in degrees C. If unknown, leave empty (press enter): ')
        salinity = input('Enter the salinity of the calibration. If unknown, leave empty (press enter): ')
        if temp == '':
            temp = defaults['temp']
        if salinity == '':
            salinity = defaults['salinity']
    else:
        temp = defaults['temp']
        salinity = defaults['salinity']
    temp = float(temp)
    salinity = float(salinity)
    
    c, rho = tsCalc.water_properties(salinity, temp, sphere_depth, lon=0.0, lat=lat)


    tsbw = {18000:{512:1750,1024:1570},38000:{512:3280,1024:2430},70000:{512:4630,1024:2830},120000:{512:5490,1024:2990},200000:{512:590,1024:3050},333000:{512:590,1024:3050}}

    if d.is_cw():
        f = d.frequency[0]
        pl = int(np.round(d.pulse_duration[0]*1000000))
        print(f, pl, materialPoperties['c1'], materialPoperties['c2'])
        fr, ts = tsCalc.freq_response(f-tsbw[f][pl]/2, f+tsbw[f][pl]/2, sphere_diameter/1000/2, c, materialPoperties['c1'], materialPoperties['c2'], rho, materialPoperties['rho1'], fstep=100)
    refTS = 10*np.log10(np.mean(10**(ts/10)))

    return refTS

def calEchogram(d_sv,f,defaults,sphereRange=None,sphereRangeTol=1):
    calview = {18000:2000,38000:2000,70000:2700,120000:5000,200000:8000,333000:8000}
    if sphereRange :
        sphereRange  = sphereRange 
    else:
        sphereRange  = sphereDict[survey][leg][sphere]
    d_sv1 = d_sv.view((0,-1,1),(0,calview[f],1))
    fig_1 = figure(figsize=(12,9))
    eg = echogram.Echogram(fig_1, d_sv1,threshold=[-90,-30])
    eg.add_colorbar(fig_1)
    plt.savefig(defaults['figure_folder']+'/'+'Echogram-'+defaults['vessel']+'-'+str(f)+'-'+np.datetime_as_string(d_sv.ping_time[0],unit='D')+'.png')
    plt.show(block=False)
    plt.pause(0.001)

    sphereRange = [i for i, x in enumerate(d_sv.depth) if np.abs(x-sphereRange )<sphereRangeTol]
    d_sv2 = d_sv.view((0,-1,1),(sphereRange[0],sphereRange[-1],1))
    fig_2 = figure(figsize=(12,3))
    eg = echogram.Echogram(fig_2, d_sv2,threshold=[-90,-30])
    eg.add_colorbar(fig_2)
    plt.show(block=False)
    plt.pause(0.001)

def detectSingleTargets(test_data,cal,detectParms,singleTargets):
    Sp = test_data.get_Sp(calibration=cal)
    along,athwart = test_data.get_physical_angles(calibration=cal)

    for ping in range(Sp.n_pings):
        cpv = 40 * np.log10(Sp.range) + 2 * cal.absorption_coefficient[ping] * Sp.range
        calPower = Sp.data[ping] - cpv
        maxima = argrelextrema(calPower, np.greater)
        

        for l in maxima[0]:
            PLDLval = calPower[l]-detectParms.PLDL
            if np.where(calPower[l:]< PLDLval)[0].size >0:
                right = l+ np.where(calPower[l:]< PLDLval)[0][0]-1
            else:
                continue

            if np.where(calPower[:l]< PLDLval)[0].size >0:
                left = np.where(calPower[:l]< PLDLval)[0][-1]+1
            else:
                continue
            
            xLeft = left+(PLDLval-calPower[left])/(calPower[left+1]-calPower[left])
            xRight = right+(PLDLval-calPower[right])/(calPower[right+1]-calPower[right])
            normWidth = (xRight - xLeft) / 4
            if (normWidth > detectParms.maxNormPulseLen) | (normWidth < detectParms.minNormPulseLen):
                continue
            
            eStartIdx = left + 1
            eEndIdx = right - 1
            if (eEndIdx - eStartIdx) < 1:
                continue
            
            peakAlong = along.data[ping][l]
            peakAthwart = athwart.data[ping][l]

            al = 2 * peakAlong / cal.beam_width_alongship[ping]
            at = 2 * peakAthwart / cal.beam_width_athwartship[ping]
            beamComp = 6.0206 * (al**2 + at**2 - (0.18 * al**2 * at**2))

            if (beamComp > detectParms.maxBeamComp):
                continue
            
            alongTarget = along.data[ping][eStartIdx:eEndIdx]
            athwartTarget = athwart.data[ping][eStartIdx:eEndIdx]

            sdAlng = np.std(alongTarget)
            if (sdAlng > detectParms.maxSDalong):
                continue
            sdAthw = np.std(athwartTarget)
            if (sdAthw > detectParms.maxSDathwart):
                continue
            r = (sum(Sp.range[eStartIdx:eEndIdx] * calPower[eStartIdx:eEndIdx])) /  sum(calPower[eStartIdx:eEndIdx]) -  (cal.sound_speed * cal.pulse_duration) / 4
            if (r > detectParms.excludeBelow) | (r < detectParms.excludeAbove):
                continue

            uTS = calPower[l] + (40 * np.log10(r)) +  (2 * cal.absorption_coefficient[ping] * r)
            cTS = uTS + beamComp
            if (cTS < detectParms.threshold_min) | (cTS > detectParms.threshold_max):
                continue

            singleTargets.ping = np.append(singleTargets.ping, ping)
            singleTargets.r = np.append(singleTargets.r, r)
            singleTargets.uTS = np.append(singleTargets.uTS, uTS)
            singleTargets.cTS = np.append(singleTargets.cTS, cTS)
            singleTargets.peakAthwart = np.append(singleTargets.peakAthwart, at)
            singleTargets.peakAlong = np.append(singleTargets.peakAlong, al)
            singleTargets.sdAlng = np.append(singleTargets.sdAlng, sdAlng)
            singleTargets.sdAthw = np.append(singleTargets.sdAthw, sdAthw)
            singleTargets.normWidth = np.append(singleTargets.normWidth, normWidth)
    
    return singleTargets

def calculate_distances(range_meters, angle_athwart_deg, angle_along_deg):
    """
    Calculate the real distances (athwart and along) from angular measurements at a given range.
    
    Parameters:
    -----------
    range_meters : float
        The range to the point in meters
    angle_athwart_deg : float
        The athwart angle in degrees (perpendicular to line of sight)
    angle_along_deg : float
        The along angle in degrees (parallel to line of sight)
    
    Returns:
    --------
    tuple
        (distance_athwart, distance_along) in meters
    """
    # Convert angles from degrees to radians
    angle_athwart_rad = np.radians(angle_athwart_deg)
    angle_along_rad = np.radians(angle_along_deg)
    
    # Calculate the distances using tangent
    distance_athwart = range_meters * np.tan(angle_athwart_rad)
    distance_along = range_meters * np.tan(angle_along_rad)
    
    return distance_athwart, distance_along


def find_points_in_sector(x_coords, y_coords, radius, sector_num=1):
    """
    Find points within a specific 1/8th sector of a circle.
    
    Parameters:
    - x_coords, y_coords: Lists of x and y coordinates
    - radius: Radius of the circle
    - sector_num: Sector number (1-8), counting counterclockwise from positive x-axis
    
    Returns:
    - mask: Boolean array indicating which points are in the sector
    - x_in_sector, y_in_sector: Arrays of coordinates in the sector
    """
    # Convert to numpy arrays if they aren't already
    x = np.array(x_coords)
    y = np.array(y_coords)
    
    # Calculate distances from origin
    distances = np.sqrt(x**2 + y**2)
    
    # Calculate angles in radians (atan2 returns values in [-π, π])
    angles = np.arctan2(y, x)
    
    # Convert to degrees and shift to [0, 360] range
    angles_deg = np.degrees(angles)
    angles_deg = (angles_deg + 360) % 360
    
    # Define sector boundaries (each sector is 45 degrees)
    sector_start = (sector_num - 1) * 45
    sector_end = sector_num * 45
    
    # Create masks for radius and angle constraints
    radius_mask = distances <= radius
    angle_mask = (angles_deg >= sector_start) & (angles_deg < sector_end)
    
    # Combine masks
    combined_mask = radius_mask & angle_mask
    
    # Get points in sector
    x_in_sector = x[combined_mask]
    y_in_sector = y[combined_mask]
    
    return combined_mask, x_in_sector, y_in_sector


def main(defaults):    
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilenames(title='Select your .raw calibration files',filetypes=[('raw files', '*.raw')])
    if isinstance(file_path, tuple):
        file_path = list(file_path)
    
    if defaults['prompt_for_params']=='True':
        sphere_diameter = input('Enter the sphere diameter in mm: ')
    else:
        sphere_diameter = defaults['sphere_diameter']
    sphere_diameter = float(sphere_diameter)
    
    if defaults['prompt_for_params']=='True':
        sphere_material = input('Enter the sphere material "Cu" or "WC": ')
    else:
        sphere_material = defaults['sphere_material']

    if sphere_material == 'Cu':
        sphere_material = 'Copper'
    elif sphere_material == 'WC':
        sphere_material = 'Tungsten carbide'

    sphere_depth = input('Enter the approximate mean sphere depth during the calibration in m: ')
    try:
        sphere_depth = float(sphere_depth)
    except:
        sphere_depth = input('Whoops, that was not a valiud number.\nEnter the approximate mean sphere depth during the calibration in m: ')
        sphere_depth = float(sphere_depth)

    sphere_depth_tol = 4

    cur_f = input('Enter the frequency of the calibration you want to look at in kHz (38 or 120): ')
    while cur_f not in ['38','120']:
        print('Invalid frequency. Please enter either 38 or 120.')
        cur_f = input('Invalid frequency. Please enter either 38 or 120: ')
    cur_f = float(cur_f)*1000

    print('Loading data...')

    ek_data = echosounder.read(file_path, frequencies=[cur_f])
    d = ek_data.get_channel_data(frequencies=[cur_f])[cur_f][0]
    cur_channel = ek_data.channel_ids[0]
    cal = echosounder.get_calibration_from_raw(ek_data)[cur_channel]

    if (d.configuration[0]['transceiver_type'] == 'GPT') & (d.configuration[0]['application_name'] == 'ES80'):
        print('This is a GPT calibration with ES software. Conducting triangle wave correction...')
        
        triwave_correcter = TriwaveCorrect(0,5)
        data, fit_results, val = triwave_correcter.triwave_correct(d)
        d_sv = data.get_Sv(calibration=cal)
        with open(defaults['figure_folder']+'/TriangleCorrection-'+defaults['vessel']+'-'+str(cur_f)+'-'+np.datetime_as_string(d_sv.ping_time[0],unit='D')+'.txt', 'w') as f:
            print(fit_results, file=f)
    else:
        d_sv = echosounder.get_Sv(ek_data,frequencies=[cur_f])[cur_channel]

    d_sv.to_depth()
 

    refTS = get_refTS(d,sphere_material,sphere_depth,sphere_diameter,defaults)


    detectParms = detectParmsInit()
    singleTargets = singleTargetsInit()

    goodSphere = 'n'
    while goodSphere == 'n':
        print('The first echogram is all of your data.')
        print('The second echogram is a zoomed in version of the sphere at the depth you previously entered.\nCHECK: Ensure you are looking at the sphere and NOT the weight.')
        calEchogram(d_sv,cur_f,defaults,sphere_depth,sphere_depth_tol)
        goodSphere = input('Does the sphere depth appear to be correct (y/n): ').lower()
        if goodSphere == 'n':
            sphere_depth = input('Enter an updated sphere depth in m: ')
            sphere_depth = float(sphere_depth)
        elif goodSphere == 'y':
            print('Continuing with the sphere depth of '+str(sphere_depth)+' m.')
        else:
            print('Invalid input. Please enter "y" or "n".')
            goodSphere = 'n'
        
        plt.close('all')

    sphere_range = sphere_depth - d_sv.depth[0]

    setattr(detectParms,'excludeAbove',sphere_range - sphere_depth_tol)
    setattr(detectParms,'excludeBelow',sphere_range + sphere_depth_tol) 


    print('Detecting single targets...this may take a while...')
    try:
        singleTargetsResults = detectSingleTargets(d,cal,detectParms,singleTargets)
    except:
        k = input('Error in detecting single targets, no targets found.\nPress enter to exit.')
        sys.exit()

    sphereHits = np.where(np.abs(singleTargetsResults.r-sphere_range)<(sphere_depth_tol/2))

    subsector_divisions = float(defaults['subsector_divisions'])

    if len(sphereHits[0]) == 0:
        print('No sphere hits were detected!')
    else:
        beam_radius_rad = np.radians(float(defaults['beam_width_deg'])/ 2)  # Half of beam width in radians
        beam_radius_meters = np.mean(singleTargetsResults.r[sphereHits])  * np.tan(beam_radius_rad)  # Radius in meters

        plt.figure(figsize=(10, 7))

        d_athwart, d_along = calculate_distances(sphere_range,singleTargetsResults.peakAthwart[sphereHits]*5,singleTargetsResults.peakAlong[sphereHits]*3.5)

        for sec in np.arange(1,9):

            mask, x,y = find_points_in_sector(d_athwart, d_along,beam_radius_meters,sector_num=sec)
            cts = np.histogram(np.sqrt(x**2+ y**2),bins=np.arange(0,beam_radius_meters+(beam_radius_meters/(subsector_divisions+1)),beam_radius_meters/subsector_divisions))[0]

            if (cts <float(defaults['min_targets_per_division'])).any():
                color='red'
            else:
                color='darkgreen'
            plt.scatter(x, y, 5, color=color, alpha=0.5, label='Random Points')
            sector_start = (sec - 1) * 45
            sector_end = sec * 45
            sector_patch = patches.Wedge((0, 0), beam_radius_meters,sector_start,sector_end,alpha=0.1,color=color)
            plt.gca().add_patch(sector_patch)

        num_onaxis = len(np.where((singleTargetsResults.peakAthwart[sphereHits]<.05)&(singleTargetsResults.peakAlong[sphereHits]<.05))[0])
        if num_onaxis <250:
            color = 'red'
        elif (num_onaxis >= 250) & (num_onaxis <500):
            color = 'yellow'
        else:
            color = 'green'
        axiscircle = Circle((0, 0), beam_radius_meters/10, color=color, linestyle='-', linewidth=1, label='Beam Outline')
        plt.gca().add_patch(axiscircle)

        beam_circle = Circle((0, 0), beam_radius_meters, fill=False, color='k', 
                            linestyle='-', linewidth=2, label='Beam Outline')
        plt.gca().add_patch(beam_circle)
        plt.axis('off')
        plt.grid()
        red_patch = patches.Patch(color='red', label='Low coverage in sector')
        yellow_patch = patches.Patch(color='yellow', label='Some coverage but not enough\n(on-axis only)')
        green_patch = patches.Patch(color='green', label='Good coverage')
        plt.legend(handles=[red_patch, yellow_patch, green_patch],bbox_to_anchor=(1, .6))
        plt.tight_layout()        
        plt.savefig(defaults['figure_folder']+'/'+'TargetsInBeam-'+defaults['vessel']+'-'+str(cur_f)+'-'+np.datetime_as_string(d_sv.ping_time[0],unit='D')+'.png')
        plt.show(block=False)
        plt.pause(0.1)

        fig = plt.figure(figsize=(8,3.5))
        plt.subplot(111)
        a = plt.hist(singleTargets.cTS[sphereHits],bins=100)
        plt.title('Single target detections in the specified range\nRed region is estimated sphere range')
        plt.fill_betweenx([0, np.max(a[0])*1.05], refTS-1.5, refTS+1.5, color='red', alpha=0.5)
        plt.ylim(0, np.max(a[0])*1.05)
        plt.grid()
        plt.savefig(defaults['figure_folder']+'/'+'TargetTS-'+defaults['vessel']+'-'+str(cur_f)+'-'+np.datetime_as_string(d_sv.ping_time[0],unit='D')+'.png')
        #plt.show(block=False)
        #plt.pause(0.1)

        esc = input('All done! Copies of all of the figures can be found in'+defaults['figure_folder']+'.\nPress enter to close all figures and exit.')
        plt.close('all')

if __name__ == "__main__": 
    sys.stdout.write('\r\n\r\n\r\n')
    sys.stdout.write('|-------------------------------------------------------------------------------|\r\n')
    sys.stdout.write("|              AVO Cal Check - Calibrate Good Times! (Come on!)                 |\r\n")
    sys.stdout.write('|-------------------------------------------------------------------------------|\r\n')
    sys.stdout.write('\r\n\r\n')

        #  read the configuration file
    try:
        config = configparser.ConfigParser()
        config.read('AVO.ini')
    except:
        #  exit with error if we can't read the config file
        sys.exit('ERROR: Unable to read configuration file: ' + configFile)
    
    defaults = dict(config.items('GENERAL'))
    main(defaults)