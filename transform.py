# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:10:42 2025

@author: Christian
"""
import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import csv
import os
import hashlib
from datetime import datetime
import numpy as np
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
import joblib
import json
import shap
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import shap


def calculate_day_of_season(df, season_start_date='12-01'):
    """
    Calculate the day of the season (starting from a given season start date, e.g., 1st December).

    Parameters:
    - df: DataFrame containing a column with date information (e.g., 'Date' in format YYYY-MM-DD).
    - season_start_date: The start date of the season in 'MM-DD' format (default is '12-01').

    Returns:
    - DataFrame with an additional 'DayOfSeason' column.
    """

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Calculate season start for each year
    season_start_year = df.index.to_series().apply(
        lambda x: x.year if x.month >= 12 else x.year - 1
    )
    season_start = pd.to_datetime(
        season_start_year.astype(str) + '-' + season_start_date
    )

    # Calculate the day of the season
    df['DayOfSeason'] = (df.index - season_start).dt.days + 1

    return df


def calculate_snow_height_differences(df):
    """Calculate snow height differences over different periods."""
    # Initialize the difference columns
    for period in [1, 2, 3, 5]:
        col_name = f'HS_delta_{period}d'
        df[col_name] = df['HSnum'].diff(periods=period)

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Identify where Stagione changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Reset differences to NaN for each new season
    for col in [f'HS_delta_{period}d' for period in [1, 2, 3, 5]]:
        df.loc[stagione_changes, col] = np.nan

    # Set HS_delta_2d, HS_delta_3d, and HS_delta_5d to NaN for 2, 3, 5 days following a Stagione change
    for period in [2, 3, 5]:
        for idx in df.index[stagione_changes]:
            # Add the specified period to the current timestamp using pd.Timedelta
            end_idx = idx + pd.Timedelta(days=period)

            # Set the snow height differences to NaN for the period following the change
            df.loc[idx + pd.Timedelta(days=1): end_idx,
                   f'HS_delta_{period}d'] = np.nan

    # Filter by z-score for each HS_delta_Xd column
    for col in [f'HS_delta_{period}d' for period in [1, 2, 3, 5]]:
        # Calculate z-scores, ignoring NaN values
        col_zscore = zscore(df[col], nan_policy='omit')
        # Set values with z-scores exceeding ±3 to NaN
        df[col] = np.where(np.abs(col_zscore) > 3, np.nan, df[col])

    return df


def calculate_new_snow(df):
    """Calculate cumulative new snow metrics over different periods."""
    df['HN_2d'] = df['HNnum'].rolling(
        window=2).sum()  # Cumulative snowfall over 2 days
    # Cumulative snowfall over 3 days
    df['HN_3d'] = df['HNnum'].rolling(window=3).sum()
    # Cumulative snowfall over 5 days
    df['HN_5d'] = df['HNnum'].rolling(window=5).sum()

    # Calculate days since last snowfall (where HNnum > 1)
    df['DaysSinceLastSnow'] = (df['HNnum'] <= 1).astype(
        int).groupby(df['HNnum'].gt(1).cumsum()).cumsum()

    return df


def calculate_temperature(df):
    """
    Calculate minimum, maximum temperatures, their differences, and temperature amplitudes over different periods.
    Resets calculations at the start of a new season ('Stagione').
    """
    # Define periods for rolling calculations and differences
    periods = [2, 3, 5]

    # Rolling min, max, and reset for each new season
    for period in periods:
        df[f'Tmin_{period}d'] = df['TminG'].rolling(window=period).min()
        df[f'Tmax_{period}d'] = df['TmaxG'].rolling(window=period).max()

    # Detect season changes and reset rolling calculations
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)
    for col in [f'Tmin_{period}d' for period in periods] + [f'Tmax_{period}d' for period in periods]:
        df.loc[stagione_changes, col] = np.nan

    # Calculate temperature amplitude
    df['TempAmplitude_1d'] = df['TmaxG'] - df['TminG']
    for period in periods:
        df[f'TempAmplitude_{period}d'] = df[f'Tmax_{period}d'] - \
            df[f'Tmin_{period}d']

    # Calculate differences for TaG, TminG, TmaxG
    for temp_col in ['TaG', 'TminG', 'TmaxG']:
        for period in [1] + periods:
            col_name = f'{temp_col}_delta_{period}d'
            df[col_name] = df[temp_col].diff(periods=period)
            df.loc[stagione_changes, col_name] = np.nan

    # Set NaN for 2, 3, 5 days after a season change for each relevant column
    for period in [2, 3, 5]:
        # Set NaN for the next 'period' days after the season change
        for idx in df.index[stagione_changes]:
            end_idx = idx + pd.Timedelta(days=period)
            # Ensure the range includes the next 'period' days
            df.loc[idx + pd.Timedelta(days=1): end_idx, [
                f'Tmin_{period}d', f'Tmax_{period}d', f'TempAmplitude_{period}d']] = np.nan
            for temp_col in ['TaG', 'TminG', 'TmaxG']:
                df.loc[idx + pd.Timedelta(days=1): end_idx,
                       f'{temp_col}_delta_{period}d'] = np.nan

    return df


def calculate_swe(df):
    """
    Calculate Snow Water Equivalent (SWE) and related precipitation metrics.

    Parameters:
    - df: DataFrame containing columns 'HNnum', 'rho', and 'Stagione'.

    Returns:
    - DataFrame with additional SWE and precipitation metrics.
    """
    # Adjust snow density based on snowfall conditions
    df['rho_adj'] = np.where(df['HNnum'] < 6, 100, df['rho'])
    df['rho_adj'] = np.where(df['HNnum'] == 0, 0, df['rho_adj'])

    # Calculate fresh snow water equivalent (FreshSWE)
    df['FreshSWE'] = df['HNnum'] * df['rho_adj'] / 100

    # Cumulative Seasonal SWE
    df['SeasonalSWE_cum'] = df.groupby('Stagione')['FreshSWE'].cumsum()

    # Rolling precipitation sums for different periods
    for period in [1, 2, 3, 5]:
        col_name = f'Precip_{period}d'
        df[col_name] = df['FreshSWE'].rolling(window=period).sum()

    # Identify season changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Set NaN for 1, 2, 3, and 5 days after a season change for precipitation sums
    for period in [1, 2, 3, 5]:
        for idx in df.index[stagione_changes]:
            end_idx = idx + pd.Timedelta(days=period)
            df.loc[idx + pd.Timedelta(days=1): end_idx,
                   [f'Precip_{p}d' for p in [1, 2, 3, 5]]] = np.nan

    # Reset rolling precipitation sums to NaN when the season changes
    for col in [f'Precip_{period}d' for period in [1, 2, 3, 5]]:
        df.loc[stagione_changes, col] = np.nan

    # Drop intermediate adjustment column
    df.drop(columns=['rho_adj'], inplace=True)

    return df


def calculate_temperature_gradient(df):
    # Calculate the temperature gradient based on snow height
    df['TempGrad_HS'] = abs(df['TH01G']) / (df['HSnum'] - 10)
    df['TempGrad_HS'] = np.where(
        df['TempGrad_HS'] == np.inf, np.nan, df['TempGrad_HS'])
    return df


def calculate_snow_temperature(df):
    """
    Calculate snow-related temperature features and categorize snow types based on temperature.
    Resets calculations when the 'Stagione' column changes.
    """
    # Hyperbolic transformations
    df['TH10_tanh'] = 20 * np.tanh(0.2 * df['TH01G'])
    df['TH30_tanh'] = 20 * np.tanh(0.2 * df['TH03G'])

    # Snow temperature differences
    for period in [1, 2, 3, 5]:
        df[f'Tsnow_delta_{period}d'] = df['TH01G'].diff(periods=period)

    # Categorize snow types based on temperature
    df['SnowConditionIndex'] = np.select(
        [df['TH01G'] < -10, (df['TH01G'] >= -10) & (df['TH01G'] < -2),
         (df['TH01G'] >= -2) & (df['TH01G'] <= 0)],
        [0, 1, 2],  # 0: Cold Snow, 1: Warm Snow, 2: Wet Snow
        default=np.nan  # Invalid condition (optional)
    )

    # Count consecutive days of wet snow
    df['ConsecWetSnowDays'] = (
        df['SnowConditionIndex'].eq(2).groupby(
            (df['SnowConditionIndex'].ne(2)).cumsum()
        ).cumsum()
    )

    # Zero out non-wet days in the consecutive count column
    df['ConsecWetSnowDays'] = np.where(
        df['SnowConditionIndex'] == 2, df['ConsecWetSnowDays'], 0)

    # Handle resets when 'Stagione' changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Set NaN for snow-related temperature features for 1, 2, 3, and 5 days after a season change
    for period in [1, 2, 3, 5]:
        # Set NaN for the following 'period' days for snow temperature differences
        for idx in df.index[stagione_changes]:
            end_idx = idx + pd.Timedelta(days=period)
            df.loc[idx + pd.Timedelta(days=1): end_idx,
                   [f'Tsnow_delta_{p}d' for p in [1, 2, 3, 5]]] = np.nan
            df.loc[idx + pd.Timedelta(days=1): end_idx, ['SnowConditionIndex',
                                                         'ConsecWetSnowDays', 'TH10_tanh', 'TH30_tanh']] = np.nan

    # Set NaN for snow-related temperature features at the start of each new season
    for col in [f'Tsnow_delta_{period}d' for period in [1, 2, 3, 5]] + [
            'SnowConditionIndex', 'ConsecWetSnowDays', 'TH10_tanh', 'TH30_tanh']:
        df.loc[stagione_changes, col] = np.nan

    return df


def calcola_stagione(data):
    anno = data.year
    if data.month >= 12:
        # stagione che inizia nell'anno corrente e termina nel successivo
        return f"{anno}/{str(anno+1)[-2:]}"
    else:
        # stagione che è iniziata l'anno precedente
        return f"{anno-1}/{str(anno)[-2:]}"
