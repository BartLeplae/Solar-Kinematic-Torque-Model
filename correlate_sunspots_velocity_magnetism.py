"""
Solar Kinematic Torque Model: Multivariate Analysis Pipeline
============================================================

A statistical and kinematic analysis pipeline designed to investigate the 
relationship between planetary orbital torque (Barycentric zero-crossings) 
and the overarching modulation of Solar Cycle amplitudes.

Core Hypothesis:
----------------
Planetary gravity accelerates the solar core, generating tangential fluid shear 
within the internal thermal dynamo. When these orbital torque events (zero-crossings) 
occur during the 'Rising Phase' of a solar cycle, they trigger premature magnetic 
reversals, constructively capping and dampening the cycle's maximum amplitude.

Data Sources:
-------------
- Sunspot Data: SIDC / SILSO (World Data Center for the Sunspot Index)
- Kinematic Data: JPL DE421 Ephemerides (via Skyfield)
- Magnetic Data: Wilcox Solar Observatory (WSO) Polar Field Data

Author: [Your Name]
Date: June 2026
License: MIT (or your chosen license)

Dependencies:
-------------
pandas, numpy, matplotlib, scipy, skyfield, drms
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load
from scipy import stats
import scipy.signal as signal
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
import re
from scipy.stats import pearsonr


def apply_zero_phase_smoothing(df, cutoff_months=13, order=4):
    """
    Applies a Butterworth low-pass filter to eliminate high-frequency unclassified noise.
    
    Unlike standard rolling averages which introduce a time-lag, a zero-phase 
    filter (via forward-backward filtering) preserves the exact temporal alignment 
    of the peaks, which is critical for precise kinematic correlation.

    Args:
        df (pd.DataFrame): The solar dataset containing 'Velocity_km_s' and 'Tangential_Acc_km_s2'.
        cutoff_months (int): The frequency threshold (in months) to filter out. Default is 13.
        order (int): The order of the Butterworth filter polynomial. Default is 4.

    Returns:
        pd.DataFrame: The original dataframe with added 'Smoothed_' columns.
    """
    print(f"\nApplying {order}th-order Butterworth low-pass filter (cutoff: {cutoff_months} months)...")
    
    # Our data is monthly, so the sampling rate is 1 observation/month
    # The Nyquist frequency is half the sampling rate
    nyquist = 0.5 
    
    # We want to filter out any signal with a period shorter than the cutoff_months
    cutoff_freq = 1.0 / cutoff_months
    
    # Normalize the frequency for SciPy's Butterworth filter design
    normal_cutoff = cutoff_freq / nyquist
    
    # Design the filter (b = numerator, a = denominator polynomials)
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Temporarily fill any potential NaNs at the edges to prevent the filter from breaking
    vel = df['Velocity_km_s'].bfill().ffill()
    acc = df['Tangential_Acc_km_s2'].bfill().ffill()
    
    # Use filtfilt (forward-backward filtering) to ensure exactly ZERO phase shift
    df['Smoothed_Velocity_km_s'] = signal.filtfilt(b, a, vel)
    df['Smoothed_Tangential_Acc'] = signal.filtfilt(b, a, acc)
    
    # We also apply it to the Sunspot data if desired, to keep signals strictly comparable
    if 'Smoothed_SSN' in df.columns:
        ssn = df['Smoothed_SSN'].bfill().ffill()
        df['Smoothed_SSN'] = signal.filtfilt(b, a, ssn)
    
    return df


def load_smoothed_sunspots(cache_file="sunspots_cache.csv", max_age_days=30):
    """Loads sunspot data from a local cache or fetches it if the cache is missing/expired."""
    
    # 1. Check if the cache exists and is fresh enough
    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age = datetime.now() - file_mod_time
        
        if age < timedelta(days=max_age_days):
            print(f"Loading data from local cache (Age: {age.days} days)...")
            # Load from CSV, ensuring our Date index is correctly parsed
            df = pd.read_csv(cache_file, index_col='Date', parse_dates=['Date'])
            return df
        else:
            print(f"Cache is {age.days} days old (expired). Fetching fresh data...")
    else:
        print("No local cache found. Fetching fresh data from SILSO...")

    # 2. Define the SILSO data endpoint and columns
    url = "https://www.sidc.be/silso/DATA/SN_ms_tot_V2.0.csv"
    cols = [
        'Year', 'Month', 'Decimal_Date', 'Smoothed_SSN', 
        'Std_Dev', 'Num_Obs', 'Definitive_Marker'
    ]
    
    # 3. Fetch and Clean the Data
    df = pd.read_csv(url, sep=r'[;,]', engine='python', header=None, names=cols)
    df['Smoothed_SSN'] = df['Smoothed_SSN'].replace(-1.0, float('nan'))
    df = df.dropna(subset=['Smoothed_SSN']).copy()
    
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df.set_index('Date', inplace=True)
    
    # 4. Save the cleaned dataframe to cache for future runs
    print(f"Saving cleaned data to '{cache_file}'...")
    df.to_csv(cache_file)
    
    return df


def append_predicted_series(df):
    """
    Appends an 'Expected_SSN' column by fitting a Hathaway curve 
    scaled dynamically to the actual maximum amplitude of each specific cycle.
    """
    # Official historical minima dates (t_0) for Solar Cycles 1 through 25
    cycle_minima = [
        pd.Timestamp('1755-02-01'), pd.Timestamp('1766-06-01'), 
        pd.Timestamp('1775-06-01'), pd.Timestamp('1784-09-01'), 
        pd.Timestamp('1798-04-01'), pd.Timestamp('1810-07-01'), 
        pd.Timestamp('1823-05-01'), pd.Timestamp('1833-11-01'), 
        pd.Timestamp('1843-07-01'), pd.Timestamp('1855-12-01'), 
        pd.Timestamp('1867-03-01'), pd.Timestamp('1878-12-01'), 
        pd.Timestamp('1890-03-01'), pd.Timestamp('1902-01-01'), 
        pd.Timestamp('1913-07-01'), pd.Timestamp('1923-08-01'), 
        pd.Timestamp('1933-09-01'), pd.Timestamp('1944-02-01'), 
        pd.Timestamp('1954-04-01'), pd.Timestamp('1964-10-01'), 
        pd.Timestamp('1976-03-01'), pd.Timestamp('1986-09-01'), 
        pd.Timestamp('1996-08-01'), pd.Timestamp('2008-12-01'), 
        pd.Timestamp('2019-12-01'), pd.Timestamp('2030-12-01')  
    ]
    
    df['Expected_SSN'] = np.nan
    
    # Iterate through each cycle block
    for i in range(len(cycle_minima) - 1):
        start_date = cycle_minima[i]
        end_date = cycle_minima[i+1]
        
        # Isolate the current cycle in the dataframe
        cycle_mask = (df.index >= start_date) & (df.index < end_date)
        cycle_data = df[cycle_mask]
        
        if len(cycle_data) == 0:
            continue
            
        # --- DYNAMIC AMPLITUDE CALCULATION ---
        # Find the highest actual smoothed sunspot number in this cycle's window.
        # If the window is empty of data (all NaNs), fallback to historical average 115.
        if cycle_data['Smoothed_SSN'].dropna().empty:
            A = 115.0
        else:
            A = cycle_data['Smoothed_SSN'].max()
        
        # Shape parameters (peak occurs at t = b * c = 3.75 years)
        b = 1.25  # Decay factor
        c = 3.0   # Rise asymmetry factor
        
        # Calculate time elapsed in months, convert to years
        months_elapsed = np.arange(len(cycle_data))
        t = months_elapsed / 12.0
        
        # Avoid division by zero
        t = np.where(t == 0, 0.001, t)
        
        # Calculate the mathematical shape
        shape = (t / b)**c * np.exp(-t / b)
        
        # Normalize so the peak of the mathematical shape equals 1
        peak_of_shape = np.max(shape) if np.max(shape) > 0 else 1
        
        # Multiply the normalized shape by our dynamically found max amplitude
        expected_values = A * (shape / peak_of_shape)
        
        # Assign back to the dataframe
        df.loc[cycle_mask, 'Expected_SSN'] = expected_values

    return df

def append_sun_kinematics(df):
    """
    Calculates the Sun's velocity and tangential acceleration relative to the 
    Solar System Barycenter for every month in the historical dataframe.
    """
    print("\nLoading extended ephemeris data (DE440)...")
    
    eph = load('de440.bsp')
    sun = eph['sun']
    ts = load.timescale()
    
    print("Calculating historical kinematics (Vectorized)...")
    
    # 1. Bypass Pandas completely
    utc_index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
    python_datetimes = utc_index.to_pydatetime().tolist()
    t = ts.from_datetimes(python_datetimes)
    
    # 2. Vectorized Velocity calculation
    state = sun.at(t)
    vel_km_s = state.velocity.km_per_s
    speed_km_s = np.linalg.norm(vel_km_s, axis=0)
    df['Velocity_km_s'] = speed_km_s
    
    # 3. Vectorized Acceleration calculation (Central Difference)
    dt_seconds = 1.0
    dt_days = dt_seconds / 86400.0
    
    t_minus = ts.tt_jd(t.tt - dt_days)
    t_plus = ts.tt_jd(t.tt + dt_days)
    
    vel_minus = sun.at(t_minus).velocity.km_per_s
    vel_plus = sun.at(t_plus).velocity.km_per_s
    
    # Raw 3D acceleration vector
    acc_km_s2 = (vel_plus - vel_minus) / (2.0 * dt_seconds)
    
    # 4. Tangential Acceleration (Dot Product)
    # First, get the unit vector of velocity (direction only)
    v_hat = vel_km_s / speed_km_s
    
    # Dot product of acceleration vector and velocity unit vector
    tangential_acc = np.sum(acc_km_s2 * v_hat, axis=0)
    
    # Append to dataframe
    df['Tangential_Acc_km_s2'] = tangential_acc
    
    return df


def plot_multivariate_solar_data(df, crossing_dates):
    """Generates a stacked 2-panel plot comparing actual sunspots and kinematics."""
    print("\nGenerating stacked multivariate historical plot...")
    
    # Create a figure with 2 subplots stacked vertically, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Reduce the horizontal space between the two boxes
    fig.subplots_adjust(hspace=0.05) 
    
    # ==========================================
    # --- TOP BOX: Sunspots ---
    # ==========================================
    color_actual = '#d35400'
    
    ax1.set_title('Solar Cycles vs. Sun Kinematics (1749 - Present)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Smoothed Sunspot Number', fontsize=12, color='black')
    
    # Plot only the Actual Sunspots
    ax1.plot(df.index, df['Smoothed_SSN'], color=color_actual, linewidth=2.0, label='Actual Smoothed SSN')
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # --- BOTTOM BOX: Kinematics ---
    # ==========================================
    color_velocity = '#27ae60'
    color_acceleration = '#8e44ad'
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Velocity relative to SSB (km/s)', fontsize=12, color=color_velocity)
    
    # Plot Velocity
    ax2.plot(df.index, df['Velocity_km_s'], color=color_velocity, linewidth=1.0, alpha=0.2)
    l3 = ax2.plot(df.index, df['Smoothed_Velocity_km_s'], color=color_velocity, linewidth=2.0, label='Smoothed Velocity')
    ax2.tick_params(axis='y', labelcolor=color_velocity)
    
    # Create a twin axis for Acceleration on the bottom box
    ax3 = ax2.twinx()
    ax3.set_ylabel('Tangential Acceleration (km/s²)', fontsize=12, color=color_acceleration)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot Acceleration
    ax3.plot(df.index, df['Tangential_Acc_km_s2'], color=color_acceleration, linewidth=1.0, alpha=0.2)
    l4 = ax3.plot(df.index, df['Smoothed_Tangential_Acc'], color=color_acceleration, linewidth=2.0, label='Smoothed Tangential Accel')
    ax3.tick_params(axis='y', labelcolor=color_acceleration)
    
    # Combined Legend for the bottom box
    lines_bottom = l3 + l4
    labels_bottom = [l.get_label() for l in lines_bottom]
    ax2.legend(lines_bottom, labels_bottom, loc='upper left', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ==========================================
    # --- EVENT MARKERS (Dashed lines in BOTH) ---
    # ==========================================
    for i, date in enumerate(crossing_dates):
        label = 'Accel Zero-Crossing' if i == 0 else ""
        
        # Add dashed line to top box
        ax1.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=1.5, label=label)
        
        # Add dashed line to bottom box
        ax2.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
        
    # Re-draw the top legend to include the vertical line label
    lines_top, labels_top = ax1.get_legend_handles_labels()
    ax1.legend(lines_top, labels_top, loc='upper left', framealpha=0.9, fontsize=11)

    plt.xlim(df.index.min(), df.index.max())
    plt.tight_layout()

    filename = f"Solar_Cycles_vs_Sun_Kinematics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    # plt.show()
    plt.close()

def analyze_zero_crossings(df):
    """
    Finds every time Tangential Acceleration crosses zero and extracts the 
    surrounding sunspot dynamics to test for cycle stagnation or delay.
    Includes a 3-month freeze period to eliminate rapid oscillation chatter.
    """
    print("\nExtracting Acceleration Zero-Crossing Events (with 3-Month Freeze)...")
    
    # 1. Calculate the rate of change (slope) of the Sunspot Number
    df['SSN_Slope'] = df['Smoothed_SSN'].diff(periods=6)
    
    # 2. Find the Raw Zero Crossings
    acc_signs = np.sign(df['Smoothed_Tangential_Acc'].dropna())
    crossings = acc_signs.diff().abs() == 2
    raw_crossing_dates = crossings[crossings].index
    
# --- 3-MONTH FREEZE DEBOUNCE LOOP ---
    filtered_crossing_dates = []
    last_date = None
    
    for date in raw_crossing_dates:
        # FIXED: Add the DateOffset to last_date before comparing timestamps
        if last_date is None or date > (last_date + pd.DateOffset(months=36)):
            filtered_crossing_dates.append(date)
            last_date = date
            
    crossing_dates = pd.DatetimeIndex(filtered_crossing_dates)

    # 3. Build a DataFrame of these specific events
    event_data = []
    for date in crossing_dates:
        # Get the exact row of data for this date
        event = df.loc[date]
        
        # Determine Cycle Phase based on SSN and its slope
        ssn = event['Smoothed_SSN']
        slope = event['SSN_Slope']
        
        if pd.isna(ssn) or pd.isna(slope):
            continue
            
        if ssn < 25.0 and abs(slope) < 5.0:
            phase = "At Minimum"
        elif slope > 0:
            phase = "Rising Phase"
        else:
            phase = "Decreasing Phase"
            
        # Calculate what happens to the slope over the NEXT 12 months
        try:
            future_date = date + pd.DateOffset(months=12)
            future_idx = df.index.get_indexer([future_date], method='nearest')[0]
            future_slope = df.iloc[future_idx]['SSN_Slope']
            slope_change = future_slope - slope
        except:
            slope_change = np.nan
            
        event_data.append({
            'Date': date.strftime('%Y-%m'),
            'Phase': phase,
            'SSN': round(ssn, 1),
            'Current_Trajectory': round(slope, 2),
            'Slope_Change_Next_12M': round(slope_change, 2)
        })
        
    events_df = pd.DataFrame(event_data)

    return crossing_dates, events_df

def extract_directional_crossings(df):
    """
    Identifies zero-crossings in tangential acceleration and mathematically 
    separates them into 'Brakes' and 'Accelerators', applying a 3-month freeze period.
    """
    print("\nExtracting Directional Zero-Crossings (with 3-Month Freeze)...")
    
    # Calculate the sign (+1 for Positive Accel, -1 for Negative Accel)
    acc_signs = np.sign(df['Smoothed_Tangential_Acc'].dropna())
    sign_changes = acc_signs.diff()
    
    # Isolate all index positions where a sign change occurred (-2 or +2)
    raw_crossings = sign_changes[sign_changes.abs() == 2]
    
    braking_dates = []
    accelerating_dates = []
    last_date = None
    
# Iterate chronologically over all raw crossings together
    for date, change in raw_crossings.items():
        # FIXED: Add the DateOffset to last_date before comparing timestamps
        if last_date is None or date > (last_date + pd.DateOffset(months=36)):
            if change == -2:
                braking_dates.append(date)
            elif change == 2:
                accelerating_dates.append(date)
            
            # Lock the freeze timer to this event
            last_date = date

    return pd.DatetimeIndex(braking_dates), pd.DatetimeIndex(accelerating_dates)

def calculate_significance(events_df):
    """
    Calculates the one-sample t-test for both Rising and Decreasing Phase stagnation.
    Applies the correct directional hypothesis (less than vs. greater than 0) based on the phase.
    """
    print("\n--- Statistical Significance (p-value) ---")
    
    # Define the phases, the expected direction of the slope change, and the t-test alternative
    phases_to_test = [
        {'phase': 'Rising Phase', 'expected_change': 'Negative', 'alt_test': 'less'},
        {'phase': 'Decreasing Phase', 'expected_change': 'Positive', 'alt_test': 'greater'}
    ]
    
    for test in phases_to_test:
        phase_name = test['phase']
        alt_test = test['alt_test']
        expected = test['expected_change']
        
        # Isolate the data for the specific phase
        phase_data = events_df[events_df['Phase'] == phase_name]['Slope_Change_Next_12M']
        
        print(f"\n[{phase_name} Stagnation]")
        
        if len(phase_data) < 2:
            print(f"Not enough data points to calculate p-value for {phase_name}.")
            continue

        # One-sample t-test: compare the mean to a population mean of 0
        t_stat, p_val = stats.ttest_1samp(phase_data, 0, alternative=alt_test)
        
        print(f"Number of events: {len(phase_data)}")
        print(f"Expected Slope Change: {expected} (Loss of momentum)")
        print(f"Actual Mean Slope Change: {phase_data.mean():.2f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4e}")
        
        if p_val < 0.05:
            print("Result: Statistically significant at the 95% confidence level (p < 0.05).")
        else:
            print("Result: Not statistically significant.")

def calculate_stratified_significance(events_df):
    """
    Calculates significance for Rising, Early Declining, and Late Declining phases.
    Splits the Declining phase based on proximity to the solar maximum (36 months).
    """
    print("\n--- Statistical Significance (Stratified) ---")
    
    # 1. Prepare segments
    rising = events_df[events_df['Phase'] == 'Rising Phase']['Slope_Change_Next_12M']
    
    # Define Early Decline as within 36 months of solar max
    # (Assuming your dataframe already has a 'Months_From_Max' column or similar)
    # If you don't have this, we can easily calculate it from your epoch data.
    early_declining = events_df[
        (events_df['Phase'] == 'Decreasing Phase') & 
        (events_df['Months_From_Max'] >= 0) & 
        (events_df['Months_From_Max'] <= 24)
    ]['Slope_Change_Next_12M']
    
    late_declining = events_df[
        (events_df['Phase'] == 'Decreasing Phase') & 
        (events_df['Months_From_Max'] > 24)
    ]['Slope_Change_Next_12M']
    
    segments = [
        ('Rising Phase', rising, 'less'),
        ('Early Decline (<24m)', early_declining, 'greater'),
        ('Late Decline (>24m)', late_declining, 'greater')
    ]
    
    for name, data, alt in segments:
        print(f"\n[Phase: {name}]")
        if len(data) < 3:
            print("Not enough data to calculate p-value.")
            continue
            
        t_stat, p_val = stats.ttest_1samp(data, 0, alternative=alt)
        print(f"Events: {len(data)} | Mean Slope: {data.mean():.2f}")
        print(f"p-value: {p_val:.4e}")
        
        if p_val < 0.05:
            print("Result: SIGNIFICANT (Stagnation detected).")
        else:
            print("Result: NOT SIGNIFICANT (System likely unresponsive).")


def plot_cycle_subplots(df, braking_dates, accelerating_dates):
    """
    Slices the historical dataset into individual solar cycles and plots them in a grid.
    Overlays the Expected Cycle and color-codes the orbital torque events.
    """
    print("\n--- Generating Color-Coded Solar Cycle Subplots ---")
    
    # 1. Automatically find the solar cycle minimums
    ssn = df['Smoothed_SSN'].dropna()
    inv_ssn = -ssn
    
    # distance=100 ensures we look for minimums at least ~8 years apart
    min_indices, _ = signal.find_peaks(inv_ssn, distance=100)
    min_dates = ssn.index[min_indices]
    
    # Create the boundaries: Start of data -> All Minimums -> End of data
    boundaries = [ssn.index[0]] + list(min_dates) + [ssn.index[-1]]
    num_cycles = len(boundaries) - 1
    
    # 2. Setup the Plot Grid
    cols = 4
    rows = int(np.ceil(num_cycles / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), sharey=True)
    axes = axes.flatten() 
    
    # 3. Plot each cycle
    for i in range(num_cycles):
        start_date = boundaries[i]
        end_date = boundaries[i+1]
        
        # Isolate the data for this specific cycle
        cycle_data = df.loc[start_date:end_date]
        ax = axes[i]
        
        # --- Plot the Sunspot Data ---
        # Expected Cycle (Background Template)
        if 'Expected_SSN' in cycle_data.columns:
            ax.plot(cycle_data.index, cycle_data['Expected_SSN'], color='#2980b9', 
                    linestyle='-', alpha=0.5, linewidth=2)
            
        # Actual Cycle (Foreground)
        ax.plot(cycle_data.index, cycle_data['Smoothed_SSN'], color='#d35400', linewidth=2.5)
        
        # --- Plot the Directional Crossings ---
        # Find crossings for this specific cycle
        cycle_brakes = [date for date in braking_dates if start_date <= date <= end_date]
        cycle_accels = [date for date in accelerating_dates if start_date <= date <= end_date]
        
        # Plot Brakes as Red (Positive to Negative)
        for date in cycle_brakes:
            ax.axvline(x=date, color='#c0392b', linestyle='--', alpha=0.8, linewidth=1.5)
            
        # Plot Accelerators as Green (Negative to Positive)
        for date in cycle_accels:
            ax.axvline(x=date, color='#27ae60', linestyle='--', alpha=0.8, linewidth=1.5)
            
        # Formatting for the subplot
        cycle_num = i if start_date.year < 1755 else i + 1 
        ax.set_title(f"Cycle {cycle_num} ({start_date.year} - {end_date.year})", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
    # 4. Hide any empty subplots in the grid 
    for j in range(num_cycles, len(axes)):
        axes[j].set_visible(False)
        
    # --- Create a Global Legend ---
    custom_lines = [
        Line2D([0], [0], color='#d35400', lw=2.5),
        Line2D([0], [0], color='#2980b9', lw=2, alpha=0.5),
        Line2D([0], [0], color='#c0392b', lw=1.5, linestyle='--'),
        Line2D([0], [0], color='#27ae60', lw=1.5, linestyle='--')
    ]
    fig.legend(custom_lines, 
               ['Actual Smoothed SSN', 'Expected Typical Cycle', 'Orbital Brake (Loss of Momentum)', 'Orbital Accelerator (Gain in Momentum)'], 
               loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.97))
               
    plt.suptitle("Solar Cycles & Directional Orbital Torques", fontsize=20, fontweight='bold', y=1.02)
    fig.text(0.02, 0.5, 'Smoothed Sunspot Number', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.94]) 
    plt.show()



def export_cycle_grid_plots(df, crossing_dates):
    """
    Slices the continuous solar timeline into individual cycles,
    arranges them in a 3x2 grid (with stacked boxes for Sunspots and Kinematics),
    and exports them as separate PNG files containing the cycle numbers.
    Event markers change color and shape based on the solar phase.
    """
    print("\nGenerating and exporting multi-cycle grids...")

    # Calculate SSN_Slope if not present to determine the phase of the zero-crossing
    if 'SSN_Slope' not in df.columns:
        df['SSN_Slope'] = df['Smoothed_SSN'].diff(periods=6)

    # Official SIDC Solar Cycle Minima (Start dates for Cycles 1 through 25)
    cycle_starts = [
        "1755-02-01", "1766-06-01", "1775-06-01", "1784-09-01", "1798-04-01", "1810-08-01",
        "1823-05-01", "1833-11-01", "1843-07-01", "1855-12-01", "1867-03-01", "1878-12-01",
        "1890-03-01", "1902-01-01", "1913-08-01", "1923-08-01", "1933-09-01", "1944-02-01",
        "1954-04-01", "1964-10-01", "1976-03-01", "1986-09-01", "1996-08-01", "2008-12-01",
        "2019-12-01"
    ]
    
    # Convert to pandas datetime
    cycle_starts = pd.to_datetime(cycle_starts)
    
    # Build a list of (start_date, end_date, cycle_number)
    cycles = []
    for i in range(len(cycle_starts)):
        start = cycle_starts[i]
        # The end of the cycle is the start of the next (or present day for Cycle 25)
        end = cycle_starts[i+1] if i + 1 < len(cycle_starts) else df.index.max()
        cycles.append((start, end, i + 1))

    # Configuration for the batches (6 cycles per file: 3 rows x 2 cols)
    cycles_per_file = 5
    num_files = int(np.ceil(len(cycles) / cycles_per_file))
    
    color_ssn = '#d35400'
    color_vel = '#27ae60'
    color_acc = '#8e44ad'

    from matplotlib.lines import Line2D  # Ensure imported for the custom legend

    for file_idx in range(num_files):
        # Determine which cycles go into this specific PNG
        batch_cycles = cycles[file_idx * cycles_per_file : (file_idx + 1) * cycles_per_file]
        
        # 1. Capture Cycle Numbers for Title and Filename
        first_cycle = batch_cycles[0][2]
        last_cycle = batch_cycles[-1][2]
        
        # Create a large high-res figure
        fig = plt.figure(figsize=(22, 16))
        fig.suptitle(f"Solar Cycles vs. Orbital Torque (Cycles {first_cycle} - {last_cycle})", 
                     fontsize=20, fontweight='bold', y=0.96)
        
        # Main Grid: 3 rows, 2 columns
        gs_main = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
        
        for idx, (start_date, end_date, cycle_num) in enumerate(batch_cycles):
            row = idx // 2
            col = idx % 2
            
            # SubGrid: 2 stacked boxes for this specific cycle cell
            gs_inner = gs_main[row, col].subgridspec(2, 1, hspace=0.05)
            
            ax1 = fig.add_subplot(gs_inner[0]) # Top Box (Sunspots)
            ax2 = fig.add_subplot(gs_inner[1], sharex=ax1) # Bottom Box (Kinematics)
            
            # Extract data for this specific cycle
            c_df = df.loc[start_date:end_date]
            if c_df.empty:
                continue
                
            c_crossings = [d for d in crossing_dates if start_date <= d <= end_date]
            
            # ==========================================
            # --- TOP BOX: Sunspots ---
            # ==========================================
            ax1.set_title(f"Solar Cycle {cycle_num} ({start_date.year} - {end_date.year})", fontsize=14, fontweight='bold')
            ax1.set_ylabel('SSN', fontsize=10)
            ax1.plot(c_df.index, c_df['Smoothed_SSN'], color=color_ssn, linewidth=2.0, label='Smoothed SSN')
            ax1.grid(True, alpha=0.3)
            
            # Hide x-axis labels for the top box to prevent clutter
            ax1.tick_params(labelbottom=False) 
            
            # ==========================================
            # --- BOTTOM BOX: Kinematics ---
            # ==========================================
            ax2.set_ylabel('Velocity (km/s)', fontsize=10, color=color_vel)
            ax2.plot(c_df.index, c_df['Smoothed_Velocity_km_s'], color=color_vel, linewidth=1.5, label='Velocity')
            ax2.tick_params(axis='y', labelcolor=color_vel)
            ax2.grid(True, alpha=0.3)
            
            # Twin axis for Acceleration
            ax3 = ax2.twinx()
            ax3.set_ylabel('Accel (km/s²)', fontsize=10, color=color_acc)
            ax3.axhline(0, color='gray', linestyle='-', alpha=0.4, linewidth=1)
            ax3.plot(c_df.index, c_df['Smoothed_Tangential_Acc'], color=color_acc, linewidth=1.5, label='Acceleration')
            ax3.tick_params(axis='y', labelcolor=color_acc)
            
            # ==========================================
            # --- EVENT MARKERS (Phase-Dependent) ---
            # ==========================================
            for d in c_crossings:
                # Default styling if data is missing
                p_color = 'black'
                p_style = '--'
                
                try:
                    event = df.loc[d]
                    ssn = event['Smoothed_SSN']
                    slope = event['SSN_Slope']
                    
                    if not (pd.isna(ssn) or pd.isna(slope)):
                        # Match the phase calculation parameters
                        if ssn < 25.0 and abs(slope) < 5.0:
                            p_color = 'gray'
                            p_style = '-.'          # At Minimum
                        elif slope > 0:
                            p_color = '#e74c3c'     # Red for Rising Phase
                            p_style = '--'
                        else:
                            p_color = '#3498db'     # Blue for Decreasing Phase
                            p_style = ':'
                except KeyError:
                    pass
                
                # Placed in both top and bottom boxes for a clean vertical slice
                ax1.axvline(x=d, color=p_color, linestyle=p_style, alpha=0.8, linewidth=2.0)
                ax2.axvline(x=d, color=p_color, linestyle=p_style, alpha=0.8, linewidth=2.0)
                
            # Add compact legends to the top right of each respective box
            ax1.legend(loc='upper right', fontsize=8, framealpha=0.8)
            
            # Combine bottom box legends
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            
            # Add proxy lines for the event legend
            event_line_rise = Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=2.0, alpha=0.8)
            event_line_dec = Line2D([0], [0], color='#3498db', linestyle=':', linewidth=2.0, alpha=0.8)
            event_line_min = Line2D([0], [0], color='gray', linestyle='-.', linewidth=2.0, alpha=0.8)
            
            ax2.legend(lines2 + lines3 + [event_line_rise, event_line_dec, event_line_min], 
                       labels2 + labels3 + ['Accel ZC (Rising)', 'Accel ZC (Declining)', 'Accel ZC (Minimum)'], 
                       loc='upper right', fontsize=8, framealpha=0.8)

        # 2. Export this batch as a highly detailed PNG with dynamic filenames
        filename = f"Solar_Cycles_{first_cycle}_to_{last_cycle}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename} ({len(batch_cycles)} cycles)")
        
        # Close the figure to free up system memory
        plt.close(fig)

    print("\nSuccess! All PNG files have been generated with phase-dependent markers in your directory.")



def calculate_kinetic_energy(df):
    """
    Calculates the Specific Kinetic Energy of the Sun using its orbital velocity.
    KE = 0.5 * v^2
    """
    print("\nCalculating Sun's Orbital Kinetic Energy...")
    
    # Check for the velocity column
    if 'Velocity_km_s' not in df.columns:
        print("[!] Missing 'Velocity_km_s' column. Cannot calculate energy.")
        return df
        
    # Calculate Specific Kinetic Energy (J/kg, assuming velocity is in km/s)
    # We square it, the 0.5 is a constant so it won't affect the normalized shape
    df['Kinetic_Energy_Raw'] = 0.5 * (df['Velocity_km_s'] ** 2)
    
    # Smooth the energy to isolate the deep currents
    df['Smoothed_Angular_Momentum'] = df['Kinetic_Energy_Raw'].rolling(window=37, center=True).mean()
    
    return df

def calculate_angular_momentum(df):
    """
    Calculates the Specific Orbital Angular Momentum (L) of the Sun 
    using the 3D cross product of Position (R) and Velocity (V).
    Handles both uppercase and lowercase column variations.
    """
    print("\nCalculating Sun's Orbital Angular Momentum...")
    
    # Dynamically map uppercase keys to the actual case used in your DataFrame
    col_map = {c.upper(): c for c in df.columns}
    required = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
    
    if not all(req in col_map for req in required):
        print("Warning: Missing 3D vector columns (X, Y, Z, VX, VY, VZ or lowercase equivalents).")
        print(f"Available columns in your dataset: {list(df.columns)}")
        return df
        
    # Extract the exact column names as they exist in your DataFrame
    x, y, z = col_map['X'], col_map['Y'], col_map['Z']
    vx, vy, vz = col_map['VX'], col_map['VY'], col_map['VZ']
    
    # Calculate the components of the cross product: L = r x v
    L_x = (df[y] * df[vz]) - (df[z] * df[vy])
    L_y = (df[z] * df[vx]) - (df[x] * df[vz])
    L_z = (df[x] * df[vy]) - (df[y] * df[vx])
    
    # Calculate the total magnitude of the Angular Momentum vector
    df['Angular_Momentum_Raw'] = np.sqrt(L_x**2 + L_y**2 + L_z**2)
    
    # Apply standard symmetric rolling mean smoothing directly here as a baseline
    df['Smoothed_Angular_Momentum'] = df['Angular_Momentum_Raw'].rolling(window=37, center=True).mean()
    
    return df

def load_wso_magnetic_data(local_file="wso_polar.txt"):
    """
    Fetches and parses the exact WSO Solar Polar Magnetic Field Data format,
    stripping out appended letters (N, S, Avg) and parsing the complex timestamp.
    """
    print("\nLoading WSO Solar Polar Magnetic Field Data...")
    
    raw_text = None
    
    # 1. Load the local file
    if os.path.exists(local_file):
        with open(local_file, 'r') as f:
            raw_text = f.read()
            
    if not raw_text or len(raw_text.strip()) == 0:
        print("[!] FATAL: The file is missing or completely empty.")
        return None
        
    # Helper function to strip letters and keep only numbers and minus signs
    def clean_num(val_str):
        return float(re.sub(r'[^\d\.-]', '', val_str))
        
    # 2. Extract Data
    data = []
    lines = raw_text.strip().split('\n')
    
    for line in lines:
        parts = line.split()
        
        # We need at least the Date, North, South, and Avg columns
        if len(parts) >= 4:
            date_str = parts[0]
            
            # Check if it starts with a year (e.g., 1976)
            if date_str[:4].isdigit():
                try:
                    # Extract just the "YYYY:MM:DD" part, ignoring the "_21h:07m:13s"
                    date_only = date_str.split('_')[0]
                    dt_index = pd.to_datetime(date_only, format='%Y:%m:%d')
                    
                    # Clean the letters off the magnetic values (e.g., "89N" -> 89.0)
                    north = clean_num(parts[1])
                    south = clean_num(parts[2])
                    avg = clean_num(parts[3])
                    
                    data.append([dt_index, north, south, avg])
                except Exception as e:
                    continue # Skip any weird header rows
                    
    if len(data) == 0:
        print("[!] The file was read, but no valid data could be parsed.")
        return None
        
    # 3. Build the DataFrame
    mag_df = pd.DataFrame(data, columns=['Date', 'North_Pole', 'South_Pole', 'Avg_Field'])
    mag_df.set_index('Date', inplace=True)
    
    # Calculate Total Dipole Strength (South is negative polarity, so we subtract)
    mag_df['Total_Dipole_Strength'] = (mag_df['North_Pole'] - mag_df['South_Pole']) / 2.0
    
    print(f"Successfully processed {len(mag_df)} magnetic field records.")
    return mag_df



def plot_unified_magnetic_kinematics(mag_df, df, braking_dates, accelerating_dates):
    """
    Creates a stacked 3-panel figure showing the complete Spin-Orbit causality chain:
    Top Box: Planetary Kinematics (Velocity & Acceleration)
    Middle Box: Internal Magnetic Field Strength
    Bottom Box: Surface Reaction (Sunspots)
    """
    if mag_df is None or len(mag_df) == 0:
        print("[!] Skipping Plot: No magnetic data available.")
        return
        
    print("\nGenerating Stacked 3-Panel Unified Plot...")
    
    # 1. Align timelines and combine events
    start_date = mag_df.index.min()
    end_date = mag_df.index.max()
    view_df = df.loc[start_date:end_date]
    
    # Safely combine DatetimeArrays by casting them to lists first
    crossing_dates = sorted(list(braking_dates) + list(accelerating_dates))
    
    # 2. Setup the Stacked Figure
    # We use a 3-row grid. The middle (Magnetics) gets slightly more height.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 1.2, 1]})
                                   
    fig.subplots_adjust(hspace=0.08) 
    
    # ==========================================
    # --- TOP BOX: Kinematics (Trigger) ---
    # ==========================================
    color_vel = '#27ae60'
    color_acc = '#8e44ad'
    
    ax1.set_title("Historical Solar Activity, Kinematics, and Magnetics (1976 - Present)", fontsize=16, fontweight='bold')
    
    # Velocity (Left Y-Axis)
    ax1.set_ylabel('Velocity (km/s)', fontsize=12, color=color_vel)
    if 'Velocity_km_s' in view_df.columns:
        ax1.plot(view_df.index, view_df['Velocity_km_s'], color=color_vel, linewidth=1.0, alpha=0.2)
    l1 = ax1.plot(view_df.index, view_df['Smoothed_Velocity_km_s'], color=color_vel, linewidth=2.0, label='Smoothed Velocity')
    ax1.tick_params(axis='y', labelcolor=color_vel)
    
    # Acceleration (Right Y-Axis)
    ax_acc = ax1.twinx()
    ax_acc.set_ylabel('Acceleration (km/s²)', fontsize=12, color=color_acc)
    if 'Tangential_Acc_km_s2' in view_df.columns:
        ax_acc.plot(view_df.index, view_df['Tangential_Acc_km_s2'], color=color_acc, linewidth=1.0, alpha=0.2)
    l2 = ax_acc.plot(view_df.index, view_df['Smoothed_Tangential_Acc'], color=color_acc, linewidth=2.0, label='Smoothed Tangential Accel')
    ax_acc.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_acc.tick_params(axis='y', labelcolor=color_acc)
    
    ax1.grid(True, alpha=0.3)
    
    # Combine legend for top box
    lines_top = l1 + l2
    labels_top = [l.get_label() for l in lines_top]
    
    # ==========================================
    # --- MIDDLE BOX: Magnetics (Engine) ---
    # ==========================================
    color_mag = '#2980b9'
    
    ax2.set_ylabel("Magnetic Field Strength", fontsize=12, color=color_mag)
    ax2.plot(mag_df.index, mag_df['North_Pole'], color='blue', alpha=0.15, label='_nolegend_')
    ax2.plot(mag_df.index, mag_df['South_Pole'], color='red', alpha=0.15, label='_nolegend_')
    
    smoothed_dipole = mag_df['Total_Dipole_Strength'].rolling(window=12, center=True).mean()
    ax2.plot(mag_df.index, smoothed_dipole, color=color_mag, linewidth=3, label='Smoothed Magnetic Field Strength')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1)
    
    ax2.tick_params(axis='y', labelcolor=color_mag)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ==========================================
    # --- BOTTOM BOX: Sunspot Cycle (Reaction) ---
    # ==========================================
    color_ssn = '#d35400'
    
    ax3.set_xlabel("Year", fontsize=12)
    ax3.set_ylabel("Smoothed Sunspot Number", fontsize=12, color='black')
    
    ax3.plot(view_df.index, view_df['Smoothed_SSN'], color=color_ssn, linewidth=2.5, label='Actual Smoothed SSN')
    
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # ==========================================
    # --- EVENT MARKERS (Dashed lines in ALL 3) ---
    # ==========================================
    crossings_in_view = [d for d in crossing_dates if start_date <= d <= end_date]
    
    for i, date in enumerate(crossings_in_view):
        label = 'Accel Zero-Crossing' if i == 0 else ""
        # Drop the vertical slice through all three panels
        ax1.axvline(x=date, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label=label)
        ax2.axvline(x=date, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        ax3.axvline(x=date, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        
    # Re-draw the top legend to include the vertical line
    handles, labels = ax1.get_legend_handles_labels()
    # We combine the velocity/acceleration handles with the newly added vertical line handle
    ax1.legend(lines_top + [handles[-1]], labels_top + ['Accel Zero-Crossing'], 
               loc='upper left', framealpha=0.9, fontsize=11)
               
    plt.xlim(start_date, end_date)
    plt.tight_layout()

    filename = f"Solar_Activity_Kinematics_Magnetics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # plt.show()
    plt.close()

def plot_continuous_cross_correlation(mag_df, df, max_lag_months=60):
    """
    Calculates and plots a 2-panel figure:
    Top: Overlapping time-series of Tangential Acceleration and Magnetic Dipole.
    Bottom: The Cross-Correlation Function (CCF) between the two signals.
    """
    print("\n--- Running Continuous Cross-Correlation Analysis ---")
    
    # 1. Normalize timelines to Monthly Start (MS) to ensure a perfect merge
    mag_monthly = mag_df[['Total_Dipole_Strength']].resample('MS').mean()
    kin_monthly = df[['Smoothed_Tangential_Acc']].resample('MS').mean()
    
    # Merge the two datasets strictly where their dates overlap (1976 - Present)
    merged = pd.merge(mag_monthly, kin_monthly, left_index=True, right_index=True, how='inner')
    merged = merged.dropna()
    
    if len(merged) < 50:
        print("[!] Not enough overlapping data points. Check data alignment.")
        return
        
    # 2. Calculate the Cross-Correlation Function (CCF)
    lags = np.arange(-max_lag_months, max_lag_months + 1)
    r_values = []
    
    for lag in lags:
        # Shift the kinematic data by 'lag' months
        shifted_kin = merged['Smoothed_Tangential_Acc'].shift(lag)
        
        temp = pd.DataFrame({
            'Mag': merged['Total_Dipole_Strength'],
            'Kin': shifted_kin
        }).dropna()
        
        r, _ = pearsonr(temp['Kin'], temp['Mag'])
        r_values.append(r)
        
    # Find the peak correlation
    best_idx = np.argmax(np.abs(r_values))
    best_lag = lags[best_idx]
    best_r = r_values[best_idx]
    
    # --- 3. Plotting the Stacked Figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.3)
    
    # ==========================================
    # --- TOP PANEL: The Raw Signals ---
    # ==========================================
    color_mag = '#2980b9'
    color_acc = '#8e44ad'
    
    ax1.set_title("Input Signals: Magnetic Dipole vs. Tangential Acceleration", fontsize=15, fontweight='bold')
    
    # Left Y-Axis: Magnetics
    ax1.set_ylabel("Total Dipole Strength", fontsize=12, color=color_mag)
    l1 = ax1.plot(merged.index, merged['Total_Dipole_Strength'], color=color_mag, linewidth=2.5, label='Magnetic Dipole Strength')
    ax1.tick_params(axis='y', labelcolor=color_mag)
    ax1.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Right Y-Axis: Acceleration
    ax1_acc = ax1.twinx()
    ax1_acc.set_ylabel("Tangential Acceleration (km/s²)", fontsize=12, color=color_acc)
    l2 = ax1_acc.plot(merged.index, merged['Smoothed_Tangential_Acc'], color=color_acc, linewidth=2.5, alpha=0.8, label='Tangential Acceleration')
    ax1_acc.tick_params(axis='y', labelcolor=color_acc)
    
    # Legend for Top Panel
    lines_top = l1 + l2
    labels_top = [l.get_label() for l in lines_top]
    ax1.legend(lines_top, labels_top, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # --- BOTTOM PANEL: The CCF ---
    # ==========================================
    ax2.set_title("Cross-Correlation Function (CCF)", fontsize=15, fontweight='bold')
    
    # Draw the CCF curve
    ax2.plot(lags, r_values, color='#e67e22', linewidth=3, label='Pearson r across time-shifts')
    
    # Highlight the peak
    ax2.axvline(best_lag, color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Peak Resonance: Lag = {best_lag} Months (r = {best_r:.3f})')
    
    # Add reference lines
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='No Time Shift (t=0)')
    
    ax2.set_xlabel("Lag Time in Months (Positive = Planetary Torque Leads Magnetic Reaction)", fontsize=12)
    ax2.set_ylabel("Pearson Correlation Coefficient (r)", fontsize=12)
    
    # Dynamically place the quadrant text based on axis limits
    y_min, y_max = ax2.get_ylim()
    text_y_pos = y_min + (y_max - y_min) * 0.1
    
    ax2.text(max_lag_months * 0.5, text_y_pos, "Planets Lead Magnetics\n(Causality)", 
             horizontalalignment='center', fontsize=11, color='green', alpha=0.7)
    ax2.text(-max_lag_months * 0.5, text_y_pos, "Magnetics Lead Planets\n(Non-Physical)", 
             horizontalalignment='center', fontsize=11, color='red', alpha=0.7)
             
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"Magnetic_dipole_vs_tangential_acceleration.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
     
    # plt.show()
    plt.close()


def plot_raw_acceleration_vs_magnetic_poles(mag_df, df, crossing_dates):
    """
    Creates a stacked 2-panel figure showing the RAW, unfiltered signals:
    Top Box: Raw Tangential Acceleration (The Trigger)
    Bottom Box: Both North and South Magnetic Poles (The Reaction)
    """
    if mag_df is None or len(mag_df) == 0:
        print("[!] Skipping Plot: No magnetic data available.")
        return
        
    print("\nGenerating Reversed Raw Signals Plot (Tangential Accel vs. Both Poles)...")
    
    # 1. Align timelines
    start_date = mag_df.index.min()
    end_date = mag_df.index.max()
    view_df = df.loc[start_date:end_date]
    
    # 2. Setup the Stacked Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.subplots_adjust(hspace=0.1) # Bring the boxes close together
    
    # ==========================================
    # --- TOP BOX: Raw Tangential Acceleration ---
    # ==========================================
    color_acc = '#8e44ad' # Violet for acceleration
    
    ax1.set_title("Raw High-Frequency Signals: Tangential Acceleration vs. Solar Magnetic Poles", fontsize=15, fontweight='bold')
    ax1.set_ylabel("Raw Tangential Accel (km/s²)", fontsize=12, color=color_acc)
    
    # Plot raw acceleration
    ax1.plot(view_df.index, view_df['Tangential_Acc_km_s2'], color=color_acc, linewidth=1.2, alpha=0.85, label='Raw Tangential Acceleration')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ==========================================
    # --- BOTTOM BOX: Both Magnetic Poles ---
    # ==========================================
    color_north = '#2980b9' # Deep blue for North Pole
    color_south = '#c0392b' # Deep red for South Pole
    
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Magnetic Field Strength", fontsize=12, color='black')
    
    # Plot raw North and South Pole data together
    ax2.plot(mag_df.index, mag_df['North_Pole'], color=color_north, linewidth=1.2, alpha=0.85, label='Raw North Pole (+)')
    ax2.plot(mag_df.index, mag_df['South_Pole'], color=color_south, linewidth=1.2, alpha=0.85, label='Raw South Pole (-)')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ==========================================
    # --- EVENT MARKERS (Dashed Lines) ---
    # ==========================================
    crossings_in_view = [d for d in crossing_dates if start_date <= d <= end_date]
    
    for i, date in enumerate(crossings_in_view):
        label = 'Accel Zero-Crossing' if i == 0 else ""
        ax1.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=1.5, label=label)
        ax2.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=1.5)

    # Add the vertical line to the top legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper left', framealpha=0.9, fontsize=11)

    plt.xlim(start_date, end_date)
    plt.tight_layout()

    filename = f"Magnetic_dipole_vs_tangential_acceleration_raw_hf.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # plt.show()
    plt.close()

def plot_raw_north_vs_acceleration(mag_df, df, crossing_dates):
    """
    Creates a stacked 2-panel figure showing the RAW, unfiltered signals:
    Top Box: Raw North Pole Magnetic Field Strength (Positive Polarity)
    Bottom Box: Raw Tangential Acceleration
    """
    if mag_df is None or len(mag_df) == 0:
        print("[!] Skipping Plot: No magnetic data available.")
        return
        
    print("\nGenerating Raw Signals Plot (North Pole vs. Tangential Accel)...")
    
    # 1. Align timelines
    start_date = mag_df.index.min()
    end_date = mag_df.index.max()
    view_df = df.loc[start_date:end_date]
    
    # 2. Setup the Stacked Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.subplots_adjust(hspace=0.1) # Bring the boxes close together
    
    # ==========================================
    # --- TOP BOX: Raw North Pole Magnetic Field ---
    # ==========================================
    color_north = '#2980b9' # Deep blue for the North Pole
    
    ax1.set_title("Raw High-Frequency Signals: North Pole Field vs. Tangential Acceleration", fontsize=15, fontweight='bold')
    ax1.set_ylabel("North Pole Field Strength", fontsize=12, color=color_north)
    
    # Plot raw North Pole data (Positive polarity)
    ax1.plot(mag_df.index, mag_df['North_Pole'], color=color_north, linewidth=1.2, alpha=0.85, label='Raw North Pole Field')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    ax1.tick_params(axis='y', labelcolor=color_north)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # --- BOTTOM BOX: Raw Tangential Acceleration ---
    # ==========================================
    color_acc = '#8e44ad' # Violet for acceleration
    
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Raw Tangential Accel (km/s²)", fontsize=12, color=color_acc)
    
    # Plot raw acceleration
    ax2.plot(view_df.index, view_df['Tangential_Acc_km_s2'], color=color_acc, linewidth=1.2, alpha=0.85, label='Raw Tangential Acceleration')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ==========================================
    # --- EVENT MARKERS (Dashed Lines) ---
    # ==========================================
    # Safely combine and filter the dates
    crossings_in_view = [d for d in crossing_dates if start_date <= d <= end_date]
    
    for i, date in enumerate(crossings_in_view):
        label = 'Accel Zero-Crossing' if i == 0 else ""
        # Drop the vertical slice through both panels
        ax1.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=1.5, label=label)
        ax2.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=1.5)

    # Add the vertical line to the top legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper left', framealpha=0.9, fontsize=11)

    plt.xlim(start_date, end_date)
    plt.tight_layout()
    plt.show()

def plot_raw_south_vs_acceleration(mag_df, df, crossing_dates):
    """
    Creates a stacked 2-panel figure showing the RAW, unfiltered signals:
    Top Box: Raw South Pole Magnetic Field Strength
    Bottom Box: Raw Tangential Acceleration
    """
    if mag_df is None or len(mag_df) == 0:
        print("[!] Skipping Plot: No magnetic data available.")
        return
        
    print("\nGenerating Raw Signals Plot (South Pole vs. Tangential Accel)...")
    
    # 1. Align timelines
    start_date = mag_df.index.min()
    end_date = mag_df.index.max()
    view_df = df.loc[start_date:end_date]
    
    # 2. Setup the Stacked Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.subplots_adjust(hspace=0.1) # Bring the boxes close together
    
    # ==========================================
    # --- TOP BOX: Raw South Pole Magnetic Field ---
    # ==========================================
    color_south = '#c0392b' # Deep red for the South Pole
    
    ax1.set_title("Raw High-Frequency Signals: South Pole Field vs. Tangential Acceleration", fontsize=15, fontweight='bold')
    ax1.set_ylabel("South Pole Field Strength", fontsize=12, color=color_south)
    
    # Plot raw South Pole data
    # Note: South pole data is typically negative, so it will mostly live below the zero line
    ax1.plot(mag_df.index, mag_df['South_Pole'], color=color_south, linewidth=1.2, alpha=0.85, label='Raw South Pole Field')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    ax1.tick_params(axis='y', labelcolor=color_south)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # --- BOTTOM BOX: Raw Tangential Acceleration ---
    # ==========================================
    color_acc = '#8e44ad' # Violet for acceleration
    
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Raw Tangential Accel (km/s²)", fontsize=12, color=color_acc)
    
    # Plot raw acceleration
    ax2.plot(view_df.index, view_df['Tangential_Acc_km_s2'], color=color_acc, linewidth=1.2, alpha=0.85, label='Raw Tangential Acceleration')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ==========================================
    # --- EVENT MARKERS (Dashed Lines) ---
    # ==========================================
    # Safely combine and filter the dates if they aren't already combined
    crossings_in_view = [d for d in crossing_dates if start_date <= d <= end_date]
    
    for i, date in enumerate(crossings_in_view):
        label = 'Accel Zero-Crossing' if i == 0 else ""
        # Drop the vertical slice through both panels
        ax1.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=1.5, label=label)
        ax2.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=1.5)

    # Add the vertical line to the top legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper left', framealpha=0.9, fontsize=11)

    plt.xlim(start_date, end_date)
    plt.tight_layout()
    plt.show()

def add_months_from_max(events_df, solar_max_dates):
    """
    Calculates the 'Months_From_Max' column by finding the absolute 
    distance between the event date and the nearest solar maximum.
    """
    def get_min_dist(row_date):
        # --- THE FIX: Convert strings to datetime objects dynamically ---
        if isinstance(row_date, str):
            row_date = pd.to_datetime(row_date)
            
        # Calculate absolute difference in months for all max dates
        distances = [abs((row_date.year - d.year) * 12 + (row_date.month - d.month)) 
                     for d in solar_max_dates]
        return min(distances)

    events_df['Months_From_Max'] = events_df['Date'].apply(get_min_dist)
    return events_df


def plot_spectral_coherence(mag_df, df, threshold=0.5):
    """
    Calculates Coherence, extracts peaks, classifies them using a 5% 
    proportional tolerance, and prints a statistical table with p-values.
    """
    if mag_df is None or len(mag_df) == 0:
        print("[!] Skipping Plot: No magnetic data available.")
        return
        
    print("\n--- Running Spectral Coherence (Fourier) Analysis ---")
    
    # 1. Normalize timelines
    mag_monthly = mag_df[['Total_Dipole_Strength']].resample('MS').mean()
    kin_monthly = df[['Smoothed_Tangential_Acc']].resample('MS').mean()
    
    merged = pd.merge(mag_monthly, kin_monthly, left_index=True, right_index=True, how='inner').dropna()
    
    if len(merged) < 100:
        print("[!] Not enough data points for a robust Fourier Transform.")
        return

    # 2. Extract arrays
    mag_sig = merged['Total_Dipole_Strength'].values
    kin_sig = merged['Smoothed_Tangential_Acc'].values
    
    # 3. Calculate Coherence
    nperseg = min(len(merged), 12 * 22) 
    frequencies, coherence = signal.coherence(kin_sig, mag_sig, fs=12, nperseg=nperseg)
    
    # Calculate segments for p-value (assuming 50% overlap)
    if len(merged) > nperseg:
        nd_segments = (len(merged) - nperseg) / (nperseg / 2) + 1
    else:
        nd_segments = 1
    
    # 4. Peak Detection
    peaks, _ = signal.find_peaks(coherence, height=threshold)
    
    astronomical_markers = [
        (11.00, "Harmonic", "Fundamental ~11-yr Solar Cycle"),
        (5.50,  "Harmonic", "1/2 Solar Cycle Harmonic"),
        (3.67,  "Harmonic", "1/3 Solar Cycle Harmonic"),
        (2.75,  "Harmonic", "1/4 Solar Cycle Harmonic"),
        (1.83,  "Harmonic", "1/6 Solar Cycle Harmonic"),
        (1.00,  "Orbital",  "Earth Orbital Period (1.00 yr)"),
        (0.615, "Orbital",  "Venus Orbital Period (~0.62 yr)"),
        (0.241, "Orbital",  "Mercury Orbital Period (~0.24 yr)"),
        (1.88,  "Orbital",  "Mars Orbital Period (~1.88 yr)"),
        (1.60,  "Synodic",  "Venus-Earth Synodic Period (~1.60 yr)"),
        (2.135, "Synodic",  "Earth-Mars Synodic Period (~2.14 yr)"),
        (1.092, "Synodic",  "Earth-Jupiter Synodic Period (~1.09 yr)"),
        (0.317, "Synodic",  "Earth-Mercury Synodic Period (~0.32 yr)"),
        (0.396, "Synodic",  "Venus-Mercury Synodic Period (~0.40 yr)")
    ]

    def classify_peak(period_yr):
        if period_yr == float('inf'):
            return "Trend", "DC component (Infinite period)"
        
        closest_match = None
        min_diff = float('inf')
        
        # Apply a proportional 5% tolerance net
        for m_val, m_type, m_desc in astronomical_markers:
            diff = abs(period_yr - m_val)
            tolerance = m_val * 0.05 # Strict 5% margin of error
            
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                closest_match = (m_type, m_desc)
                
        return closest_match if closest_match else ("Unclassified", "Complex interaction / Sideband Noise")

    # Build peak data
    peak_data = []
    for p in peaks:
        freq = frequencies[p]
        coh = coherence[p]
        period_years = 1 / freq if freq > 0 else float('inf')
        period_months = period_years * 12
        p_val = (1.0 - coh) ** (nd_segments - 1) if nd_segments > 1 else 1.0
        
        p_type, p_comment = classify_peak(period_years)
        
        peak_data.append({
            'period_yr': period_years,
            'period_mo': period_months,
            'coherence': coh,
            'p_value': p_val,
            'type': p_type,
            'comment': p_comment
        })

    # Filter out periods shorter than your 1.5-year (18 month) cutoff
    peak_data = [row for row in peak_data if row['period_yr'] >= 0.5]

    # Sort the peaks by Period (Longest to Shortest)
    peak_data.sort(key=lambda x: x['period_yr'], reverse=True)


    # Write Markdown file
    with open('coherence_results.md', 'w') as f:
        f.write("### Significant Resonant Peaks\n\n")
        f.write("| Period (Yrs) | Coherence (r²) | p-value | Classification | Comment |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for d in peak_data:
            f.write(f"| {d['period_yr']:.2f} | {d['coherence']:.4f} | {d['p_value']:.4f} | {d['type']} | {d['comment']} |\n")

    print("\n[✓] Results exported:'coherence_results.md'")


def plot_sun_barycenter_magnetic_trajectory():
    print("Loading WSO Magnetic Data...")
    mag_df = magnetic_data
    
    # Resample to monthly to match standard orbital smoothing
    mag_monthly = mag_df[['Total_Dipole_Strength', 'North_Pole', 'South_Pole']].resample('MS').mean().dropna()
    
    print("Calculating Sun's position relative to the Solar System Barycenter...")
    # Load Skyfield Ephemeris
    eph = load('de421.bsp')
    sun = eph['sun']
    # ssb = eph['solar system barycenter']
    ts = load.timescale()
    
    # Create timescales matching your magnetic data dates
    # FIX: Localize to UTC, then convert to pure Python datetimes so Skyfield can mutate the array
    dates = mag_monthly.index.tz_localize('UTC').to_pydatetime()
    # FIX: Use from_datetimes (plural) for an array of dates!
    t = ts.from_datetimes(dates)
    
    # Get the pure, instantaneous geometric position of the Sun relative to the Barycenter
    pos_au = sun.at(t).position.au
    x, y = pos_au[0], pos_au[1]  # Extracting X and Y coordinates on the ecliptic plane

    # Add coordinates back to our dataframe
    mag_monthly['x'] = x
    mag_monthly['y'] = y
    
    # Determine the polarity (Direction of the overarching magnetic field)
    mag_monthly['Polarity'] = np.sign(mag_monthly['North_Pole'])
    
    # --- PREPARE DATA FOR VARIABLE LINE PLOTTING ---
    # Slicing the continuous path into segments (x, y coordinate pairs)
    points = np.array([mag_monthly['x'].values, mag_monthly['y'].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 1. Map 'Total_Dipole_Strength' to Line Width
    widths = mag_monthly['Total_Dipole_Strength'].values[:-1]
    min_w, max_w = np.nanmin(widths), np.nanmax(widths)
    # Normalize widths to a visible pixel range (e.g., 1 to 8)
    norm_widths = 1 + 7 * (widths - min_w) / (max_w - min_w)
    
    # 2. Map 'Polarity' to Color (Red/Blue)
    colors = ['#e74c3c' if pol > 0 else '#3498db' for pol in mag_monthly['Polarity'].values[:-1]]
    
    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#f8f9fa')
    
    # Plot the Barycenter at (0,0)
    ax.plot(0, 0, marker='+', color='black', markersize=20, markeredgewidth=2, label='Solar System Barycenter (SSB)')
    
    # Add the colored, variable-width trajectory
    lc = LineCollection(segments, linewidths=norm_widths, colors=colors, alpha=0.8, capstyle='round')
    ax.add_collection(lc)
    
    # Formatting the visual aesthetics
    ax.set_xlim(mag_monthly['x'].min() * 1.1, mag_monthly['x'].max() * 1.1)
    ax.set_ylim(mag_monthly['y'].min() * 1.1, mag_monthly['y'].max() * 1.1)
    ax.set_aspect('equal') # Forces the X and Y axis to use the same physical scale (true circles)
    
    ax.set_xlabel("X Position (Astronomical Units)", fontsize=12)
    ax.set_ylabel("Y Position (Astronomical Units)", fontsize=12)
    ax.set_title("The Sun's Barycentric Orbit & Magnetic Field\n(Width = Dipole Strength | Color = Magnetic Polarity)", fontsize=16, fontweight='bold')
    
    # Custom Graphic Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#e74c3c', lw=5, label='Positive Polarity (North +)'),
        Line2D([0], [0], color='#3498db', lw=5, label='Negative Polarity (North -)'),
        Line2D([0], [0], color='gray', lw=8, label='Strong Dipole (Solar Minimum)'),
        Line2D([0], [0], color='gray', lw=2, label='Weak Dipole (Reversal / Solar Max)')
    ]
    ax.legend(handles=custom_lines, loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    
    # Draw reference rings 
    circle_01 = plt.Circle((0, 0), 0.005, color='gray', fill=False, linestyle=':', alpha=0.5)
    circle_02 = plt.Circle((0, 0), 0.010, color='gray', fill=False, linestyle=':', alpha=0.5)
    ax.add_patch(circle_01)
    ax.add_patch(circle_02)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    filename = f"Sun_Barycentric_Orbit_and_Magnetic_Field.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # plt.show()
    plt.close()

def test_reversal_phase_amplitude(cycles_df):
    """
    Tests if solar cycles with a magnetic reversal during the rising phase 
    have a significantly lower amplitude than those with a reversal in the declining phase.
    """
    print("\n--- Statistical Test: Reversal Phase vs. Cycle Amplitude ---")
    
    # 1. Isolate the two groups based on when the reversal happened
    rising_reversals = cycles_df[cycles_df['Phase'] == 'Rising']['Amplitude']
    declining_reversals = cycles_df[cycles_df['Phase'] == 'Declining']['Amplitude']
    
    # 2. Check if we have enough data points to run the math
    if len(rising_reversals) < 2 or len(declining_reversals) < 2:
        print("Error: Not enough data points in one or both groups to perform a t-test.")
        return None, None
        
    # 3. Print Descriptive Statistics
    print(f"[Reversal during Rising Phase]")
    print(f"Count: {len(rising_reversals)} cycles | Mean Amplitude: {rising_reversals.mean():.1f}")
    
    print(f"\n[Reversal during Declining Phase]")
    print(f"Count: {len(declining_reversals)} cycles | Mean Amplitude: {declining_reversals.mean():.1f}")
    
    # 4. Perform Welch's T-Test (One-Sided)
    # We use alternative='less' because the hypothesis is that Rising < Declining.
    # equal_var=False uses Welch's t-test, which is safer for unequal sample sizes.
    t_stat, p_value = stats.ttest_ind(
        rising_reversals, 
        declining_reversals, 
        alternative='less', 
        equal_var=False
    )
    
    print(f"\n--- T-Test Results ---")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_value:.4e}")
    
    # 5. Conclusion
    alpha = 0.05
    if p_value < alpha:
        print("\nResult: SIGNIFICANT.")
        print("Conclusion: Cycles that reverse during the rising phase have a statistically confirmed LOWER amplitude.")
    else:
        print("\nResult: NOT SIGNIFICANT.")
        print("Conclusion: There is not enough statistical evidence to prove that a rising phase reversal dampens the cycle's amplitude.")
        
    return t_stat, p_value

def test_historic_kinematic_reversals(df, crossing_dates):
    """
    Extracts historical solar cycles and tests the Phase Disruption Hypothesis.
    
    Isolates each solar cycle using its sunspot minimums, identifies the first 
    kinematic zero-crossing within that cycle, and classifies whether the 
    torque event occurred during the 'Rising' or 'Declining' phase. Results 
    are exported to a Markdown table for documentation.

    Args:
        df (pd.DataFrame): The dataset containing 'Smoothed_SSN'.
        crossing_dates (list): Datetime indexes of all kinematic zero-crossings.

    Returns:
        pd.DataFrame: A formatted table of historical cycles and their phase classifications.
    """
    print("\n--- Building Historic Cycle Summary (Kinematic Proxy) ---")
    
    # 1. Automatically Identify Solar Cycle Boundaries (Minimums)
    # By inverting the SSN, we can use find_peaks to locate the valleys.
    # A distance of 100 months (~8.3 years) ensures we don't double-count noisy minimums.
    ssn_clean = df['Smoothed_SSN'].dropna()
    min_indices, _ = signal.find_peaks(-ssn_clean, distance=100)
    cycle_min_dates = ssn_clean.iloc[min_indices].index
    
    cycle_data = []
    
    # 2. Iterate through historic cycles (from one minimum to the next)
    for i in range(len(cycle_min_dates) - 1):
        start_date = cycle_min_dates[i]
        end_date = cycle_min_dates[i+1]
        
        # Isolate the data for just this one cycle
        cycle_slice = ssn_clean.loc[start_date:end_date]
        
        # Find the Peak (Amplitude) and when it occurred
        max_ssn = cycle_slice.max()
        max_date = cycle_slice.idxmax()
        
        # Find all kinematic crossings that happened inside this cycle's window
        crossings_in_cycle = [d for d in crossing_dates if start_date <= d < end_date]
        
        if not crossings_in_cycle:
            continue  # Skip if no crossing happened
            
        # The first crossing triggers the internal disruption / phase shift
        first_crossing = crossings_in_cycle[0]
        
        # Classify the Phase
        if first_crossing < max_date:
            phase = 'Rising'
        else:
            phase = 'Declining'
            
        cycle_data.append({
            'Cycle_Index': i + 1,
            'Start_Date': start_date.strftime('%Y-%m'),
            'Max_Date': max_date.strftime('%Y-%m'),
            'Amplitude': round(max_ssn, 1),
            'Kinematic_Reversal': first_crossing.strftime('%Y-%m'),
            'Reversal_Phase': phase
        })
        
    cycles_df = pd.DataFrame(cycle_data)

    filename = 'historic_kinematic_reversals.md'
    with open(filename, 'w') as f:
        f.write("# Historic Kinematic Solar Cycle Summary\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("| Cycle | Start | Max | Amplitude | Kinematic Event | Phase |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for _, row in cycles_df.iterrows():
            f.write(f"| {row['Cycle_Index']} | {row['Start_Date']} | {row['Max_Date']} | "
                    f"{row['Amplitude']} | {row['Kinematic_Reversal']} | {row['Reversal_Phase']} |\n")
                    
    print(f"[✓] Analysis complete. Data exported to '{filename}'") 

    print(f"Successfully mapped {len(cycles_df)} historic solar cycles.")

    # 3. Run the Statistical T-Test on the massive historical dataset
    print("\n--- Statistical Test: Kinematic Proxy vs. Cycle Amplitude ---")
    
    rising_reversals = cycles_df[cycles_df['Reversal_Phase'] == 'Rising']['Amplitude']
    declining_reversals = cycles_df[cycles_df['Reversal_Phase'] == 'Declining']['Amplitude']
    
    if len(rising_reversals) < 2 or len(declining_reversals) < 2:
        print("Error: Not enough data points to run t-test.")
        return cycles_df
        
    # Welch's T-Test (One-Sided: hypothesis is Rising < Declining)
    t_stat, p_value = stats.ttest_ind(
        rising_reversals, 
        declining_reversals, 
        alternative='less', 
        equal_var=False
    )

    # Append Statistics to the Markdown file
    with open(filename, 'a') as f:
        f.write(f"\n## Statistical Test: Rising vs. Declining Phase\n")
        f.write(f"| Metric | Rising Phase | Declining Phase |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| **Count** | {len(rising_reversals)} | {len(declining_reversals)} |\n")
        f.write(f"| **Mean Amplitude** | {rising_reversals.mean():.1f} | {declining_reversals.mean():.1f} |\n\n")
        
        f.write(f"### T-Test Validation\n")
        f.write(f"- **T-Statistic:** `{t_stat:.4f}`\n")
        f.write(f"- **P-Value:** `{p_value:.4e}`\n")
        f.write(f"- **Note:** Welch's one-sided test (Rising < Declining).\n")

    # Calculate Statistical Significance using a Permutation Test
    n_permutations = 10000
    res = stats.permutation_test(
        (declining_reversals, rising_reversals), # Note: using your previous variable names
        statistic=lambda x, y: np.mean(x) - np.mean(y), 
        permutation_type='independent', 
        alternative='greater', 
        n_resamples=n_permutations
    )

    # Append Permutation Statistics to the Markdown file
    with open(filename, 'a') as f:
        f.write(f"\n## Advanced Validation: Permutation Test\n")
        f.write(f"To eliminate reliance on bell-curve assumptions, we conducted a non-parametric permutation test.\n\n")
        f.write(f"- **Resamples:** {n_permutations:,}\n")
        f.write(f"- **Observed Difference in Means:** {res.statistic:.2f} sunspots\n")
        f.write(f"- **Empirical P-Value:** `{res.pvalue:.5f}`\n\n")
        
        # Add interpretation
        if res.pvalue < 0.01:
            f.write(f"**Conclusion:** The result is highly significant. It is mathematically improbable (p < 0.01) "
                    f"that the observed amplitude suppression in the Rising Phase occurred by random chance.")

    return cycles_df

def plot_reversal_amplitudes_stacked(cycles_df):
    """
    Visualizes the statistical impact of phase-dependent torque on cycle amplitude.
    
    Generates a dual-layout plot featuring a jittered boxplot and vertically stacked 
    histograms. Calculates and annotates statistical significance dynamically using 
    a 10,000-resample Permutation Test.

    Args:
        cycles_df (pd.DataFrame): The output table from `test_historic_kinematic_reversals`.

    Outputs:
        Saves 'Impact_of_reversal_on_solar_cycle_amplitude.png' to local directory.
    """
    print("\nGenerating Stacked Amplitude vs. Phase Visualization...")
    
    # Isolate the data and calculate N
    rising = cycles_df[cycles_df['Reversal_Phase'] == 'Rising']['Amplitude'].dropna()
    declining = cycles_df[cycles_df['Reversal_Phase'] == 'Declining']['Amplitude'].dropna()
    
    mean_rise = rising.mean()
    mean_dec = declining.mean()
    n_rise = len(rising)
    n_dec = len(declining)

    # Calculate Statistical Significance using a Permutation Test
    n_permutations = 10000
    res = stats.permutation_test(
        (declining, rising), 
        statistic=lambda x, y: np.mean(x) - np.mean(y), 
        permutation_type='independent', 
        alternative='greater', 
        n_resamples=n_permutations
    )

    # --- Generate the Automated Interpretation ---
    if res.pvalue < 0.01:
        significance = "HIGHLY SIGNIFICANT"
        conclusion = "Premature kinematic disruption actively\ncaps and suppresses solar cycle amplitude."
    elif res.pvalue < 0.05:
        significance = "SIGNIFICANT"
        conclusion = "Premature kinematic disruption limits\nsolar cycle amplitude."
    else:
        significance = "NOT SIGNIFICANT"
        conclusion = "Insufficient statistical evidence to prove\nkinematic suppression."

    # Create a custom Grid Layout
    fig = plt.figure(figsize=(14, 8), facecolor='#f8f9fa')
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # ==========================================
    # --- SUBPLOT 1 (LEFT): JITTERED BOXPLOT ---
    # ==========================================
    ax_box = fig.add_subplot(gs[:, 0]) 
    
    # ---> FIX: Changed 'labels' to 'tick_labels' here to resolve the Matplotlib 3.9+ deprecation warning <---
    box = ax_box.boxplot([rising, declining], 
                     tick_labels=['Rising Phase\n(Premature Disruption)', 'Declining Phase\n(Natural Reversal)'],
                     patch_artist=True,
                     widths=0.5,
                     medianprops=dict(color='black', linewidth=2.5),
                     flierprops=dict(marker='o', color='black', alpha=0.5))
    
    colors = ['#e74c3c', '#3498db']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    # Jittered Scatter Overlay
    ax_box.scatter(np.random.normal(1, 0.05, size=n_rise), rising, 
               color='darkred', edgecolor='white', s=50, alpha=0.9, zorder=3)
    ax_box.scatter(np.random.normal(2, 0.05, size=n_dec), declining, 
               color='darkblue', edgecolor='white', s=50, alpha=0.9, zorder=3)

    ax_box.set_title('Statistical Distribution (Boxplot)', fontsize=14, fontweight='bold')
    ax_box.set_ylabel('Maximum Sunspot Number (Amplitude)', fontsize=12, fontweight='bold')
    ax_box.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Add the Upgraded Statistical Text Box ---
    stats_text = (f"Test: Permutation Resampling ({n_permutations:,} runs)\n"
                  f"Hypothesis: Rising phase disruption lowers amplitude.\n"
                  f"{'-'*45}\n"
                  f"Observed Difference: {res.statistic:.1f} sunspots\n"
                  f"Empirical P-Value: {res.pvalue:.5f}\n"
                  f"{'-'*45}\n"
                  f"Result: {significance}\n"
                  f"Conclusion: {conclusion}")
                  
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray')
    
    # Anchored to the top left of the subplot (slightly smaller font to fit the extra text)
    ax_box.text(0.04, 0.96, stats_text, transform=ax_box.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left', bbox=props, fontweight='bold', zorder=5)

    # ==========================================
    # --- CALCULATE SHARED BINS ---
    # ==========================================
    bins = np.linspace(min(rising.min(), declining.min()) - 10, 
                       max(rising.max(), declining.max()) + 10, 12)

    # ==========================================
    # --- SUBPLOT 2 (TOP RIGHT): RISING PHASE ---
    # ==========================================
    ax_rise = fig.add_subplot(gs[0, 1]) 
    
    ax_rise.hist(rising, bins=bins, color='#e74c3c', alpha=0.8, edgecolor='white', label=f'Cycles (N={n_rise})')
    ax_rise.axvline(mean_rise, color='darkred', linestyle='dashed', linewidth=2.5, label=f'Mean ({mean_rise:.0f})')
    
    ax_rise.set_title('Population Density: Rising Phase Disruption', fontsize=12, fontweight='bold')
    ax_rise.set_ylabel('Number of Cycles', fontsize=10, fontweight='bold')
    ax_rise.grid(axis='y', linestyle='--', alpha=0.7)
    ax_rise.legend(loc='upper right')
    ax_rise.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ==========================================
    # --- SUBPLOT 3 (BOTTOM RIGHT): DECLINING PHASE ---
    # ==========================================
    ax_dec = fig.add_subplot(gs[1, 1], sharex=ax_rise) 
    
    ax_dec.hist(declining, bins=bins, color='#3498db', alpha=0.8, edgecolor='white', label=f'Cycles (N={n_dec})')
    ax_dec.axvline(mean_dec, color='darkblue', linestyle='dashed', linewidth=2.5, label=f'Mean ({mean_dec:.0f})')
    
    ax_dec.set_title('Population Density: Natural Declining Reversal', fontsize=12, fontweight='bold')
    ax_dec.set_xlabel('Maximum Sunspot Number (Amplitude)', fontsize=12, fontweight='bold')
    ax_dec.set_ylabel('Number of Cycles', fontsize=10, fontweight='bold')
    ax_dec.grid(axis='y', linestyle='--', alpha=0.7)
    ax_dec.legend(loc='upper right')
    ax_dec.yaxis.set_major_locator(MaxNLocator(integer=True))

    # --- FINAL FORMATTING ---
    plt.suptitle('Impact of Kinematic Reversal Phase on Solar Cycle Amplitude', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    filename = f"Impact_of_reversal_on_solar_cycle_amplitude.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # plt.show()
    plt.close()

if __name__ == "__main__":

    # 1. Fetch or load the cached sunspot data
    historical_sunspots = load_smoothed_sunspots()
    
    # 2. Append the predicted Hathaway curve
    historical_sunspots = append_predicted_series(historical_sunspots)
    
    # 3. Append the Solar System Barycenter kinematics
    historical_sunspots = append_sun_kinematics(historical_sunspots)


# 2. Calculate Angular Momentum (BEFORE smoothing everything else)
    # historical_sunspots = calculate_angular_momentum(historical_sunspots)
    historical_sunspots = calculate_kinetic_energy(historical_sunspots)

# --- FAILSAFE: Backup the angular momentum columns ---
    saved_momentum_cols = {}
    for col in ['Angular_Momentum_Raw', 'Smoothed_Angular_Momentum']:
        if col in historical_sunspots.columns:
            saved_momentum_cols[col] = historical_sunspots[col]

    # 4. Apply Zero-Phase Smoothing to the kinematics
    # historical_sunspots = apply_zero_phase_smoothing(historical_sunspots, window=80)
    historical_sunspots = apply_zero_phase_smoothing(historical_sunspots, cutoff_months=18)



# --- FAILSAFE: Restore columns if they were dropped by the smoothing step ---
    for col, series in saved_momentum_cols.items():
        historical_sunspots[col] = series

    # 2. Extract Events and Test Hypothesis
    crossing_dates, events_df = analyze_zero_crossings(historical_sunspots)

# --- Define your solar max dates (Update these to your specific cycle peaks) ---
    # Official SIDC/SILSO Solar Cycle Maximums (Based on 13-month smoothed sunspot number)
    solar_max_dates = [
        pd.Timestamp('1761-06-01'),  # Cycle 1
        pd.Timestamp('1769-09-01'),  # Cycle 2
        pd.Timestamp('1778-05-01'),  # Cycle 3
        pd.Timestamp('1788-02-01'),  # Cycle 4
        pd.Timestamp('1805-02-01'),  # Cycle 5
        pd.Timestamp('1816-05-01'),  # Cycle 6
        pd.Timestamp('1829-11-01'),  # Cycle 7
        pd.Timestamp('1837-03-01'),  # Cycle 8
        pd.Timestamp('1848-02-01'),  # Cycle 9
        pd.Timestamp('1860-02-01'),  # Cycle 10
        pd.Timestamp('1870-08-01'),  # Cycle 11
        pd.Timestamp('1883-12-01'),  # Cycle 12
        pd.Timestamp('1894-01-01'),  # Cycle 13
        pd.Timestamp('1906-02-01'),  # Cycle 14
        pd.Timestamp('1917-08-01'),  # Cycle 15
        pd.Timestamp('1928-04-01'),  # Cycle 16
        pd.Timestamp('1937-04-01'),  # Cycle 17
        pd.Timestamp('1947-05-01'),  # Cycle 18
        pd.Timestamp('1957-03-01'),  # Cycle 19
        pd.Timestamp('1968-11-01'),  # Cycle 20
        pd.Timestamp('1979-12-01'),  # Cycle 21
        pd.Timestamp('1989-11-01'),  # Cycle 22
        pd.Timestamp('2001-11-01'),  # Cycle 23
        pd.Timestamp('2014-04-01'),  # Cycle 24
        pd.Timestamp('2024-10-01')   # Cycle 25 (Latest consensus peak)
    ]

    # --- Usage before calling the stratified test ---
    events_df = add_months_from_max(events_df, solar_max_dates)
    # calculate_stratified_significance(events_df)

    # 3. Plot the data with event markers
    plot_multivariate_solar_data(historical_sunspots, crossing_dates)

# 3. Generate the Grid Plot
    # plot_cycle_subplots(historical_sunspots, crossing_dates)


# 2. Extract the Directional Zero-Crossings
    braking_dates, accelerating_dates = extract_directional_crossings(historical_sunspots)

    export_cycle_grid_plots(historical_sunspots, crossing_dates)


# 2. Fetch the Raw Magnetic Data
    magnetic_data = load_wso_magnetic_data()

    plot_sun_barycenter_magnetic_trajectory()
    

# 5. Plot the Unified Master Stack
    plot_unified_magnetic_kinematics(magnetic_data, historical_sunspots, braking_dates, accelerating_dates)

# Run the continuous wave correlation
    plot_continuous_cross_correlation(magnetic_data, historical_sunspots)


    crossing_dates = sorted(list(braking_dates) + list(accelerating_dates))
# Run the historic statistical test!
    historic_cycles_df = test_historic_kinematic_reversals(historical_sunspots, crossing_dates)

# 3. Plot the Boxplot!
    if historic_cycles_df is not None and not historic_cycles_df.empty:
        plot_reversal_amplitudes_stacked(historic_cycles_df)

    # Generate the raw data comparison plot
    plot_raw_acceleration_vs_magnetic_poles(magnetic_data, historical_sunspots, crossing_dates)

    # plot_high_frequency_residuals(magnetic_data, historical_sunspots)

    plot_spectral_coherence(magnetic_data, historical_sunspots)

# Save your findings for further analysis
    events_df.to_csv("solar_cycle_events_analysis.csv", index=False)
    print("\nAnalysis saved to 'solar_cycle_events_analysis.csv'.")

