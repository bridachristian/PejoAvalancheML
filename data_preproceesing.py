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


# URL dei dati
URL = "http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve"
CODICE_STAZIONE = "21MB"
FILE_CSV = "storico_21MB.csv"
HASH_FILE = "last_hash.txt"   # qui salvo l’hash dell’ultimo XML scaricato


def fetch_xml():
    """Scarica il file XML"""
    r = requests.get(URL)
    r.raise_for_status()
    return r.content


def xml_changed(xml_data):
    """Controlla se il contenuto è diverso dal file precedente"""
    new_hash = hashlib.md5(xml_data).hexdigest()

    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            old_hash = f.read().strip()
        if new_hash == old_hash:
            print("⚠️ Nessun cambiamento rispetto a ieri, esco.")
            return False

    # aggiorno hash salvato
    with open(HASH_FILE, "w") as f:
        f.write(new_hash)

    return True


def parse_xml(xml_data, codStaz):
    """Estrae i rilievi neve della stazione con codStaz"""
    root = ET.fromstring(xml_data)
    rilievi = []
    # Attenzione: il file ha namespace xmlns="http://www.meteotrentino.it/"
    ns = {"ns": "http://www.meteotrentino.it/"}
    for ril in root.findall("ns:rilievo_neve", ns):
        if ril.findtext("ns:codStaz", namespaces=ns) == codStaz:
            rilievi.append({
                "DataRilievo": ril.findtext("ns:dataMis", namespaces=ns),
                # "OraRilievo": ril.findtext("ns:oraDB", namespaces=ns),
                "Ta": ril.findtext("ns:ta", namespaces=ns),
                "Tmin": ril.findtext("ns:tmin", namespaces=ns),
                "Tmax": ril.findtext("ns:tmax", namespaces=ns),
                "HS": ril.findtext("ns:hs", namespaces=ns),
                "HN": ril.findtext("ns:hn", namespaces=ns),
                "rho": ril.findtext("ns:fi", namespaces=ns),
                "TH10": ril.findtext("ns:t10", namespaces=ns),
                "TH30": ril.findtext("ns:t30", namespaces=ns),
                "PR": ril.findtext("ns:pr", namespaces=ns)
            })
    return rilievi


def convert_rilievo(r):
    """Converte i valori stringa di un rilievo in numerici + DataRilievo in datetime"""
    def to_num(val):
        if val is None:
            return None
        v = val.strip()
        if v in ["///", "//", "", None]:
            return np.nan
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return np.nan

    def to_date(val):
        if not val:
            return np.nan
        try:
            # formato visto nell'XML: '21/09/2025 00:00:00'
            return datetime.strptime(val.strip(), "%d/%m/%Y %H:%M:%S")
        except ValueError:
            return np.nan

    return {
        "DataRilievo": to_date(r["DataRilievo"]),
        "Ta": to_num(r["Ta"]),
        "Tmin": to_num(r["Tmin"]),
        "Tmax": to_num(r["Tmax"]),
        "HS": to_num(r["HS"]),
        "HN": to_num(r["HN"]),
        "rho": to_num(r["rho"]),
        "TH10": to_num(r["TH10"]),
        "TH30": to_num(r["TH30"]),
        "PR": to_num(r["PR"])
    }


def convert_all_rilievi(rilievi):
    """Converte una lista di rilievi in una tabella pandas"""
    return pd.DataFrame([convert_rilievo(r) for r in rilievi])


def converti_aineva(df):
    """
    Converte un DataFrame secondo le specifiche AINEVA:
    - Temperature >50 diventano negative con suffisso 'G'
    - Sostituisce None in 'rho' con 0
    - Assicura che HS, HN, PR siano >=0
    - Verifica che Tmin < Tmax e che Ta sia compresa tra Tmin e Tmax
    """

    def convert_temperatures(df):
        temp_cols = ['Ta', 'Tmin', 'Tmax', 'TH10', 'TH30']
        for col in temp_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mask = df[col] >= 50
                if mask.any():
                    df.loc[mask, col] = -(df.loc[mask, col] - 50)
                    # df.rename(columns={col: col + 'G'}, inplace=True)
        return df

    def fix_temperature_order(df):
        if all(c in df.columns for c in ['Tmin', 'Ta', 'Tmax']):
            for i, row in df.iterrows():
                trio = [row['Tmin'], row['Ta'], row['Tmax']]
                if any(pd.isna(trio)):
                    continue  # salta righe con NaN
                sorted_vals = sorted(trio)
                # assegna in ordine corretto
                df.at[i, 'Tmin'], df.at[i, 'Ta'], df.at[i, 'Tmax'] = sorted_vals
        return df

    def convert_rho(df):
        if 'rho' in df.columns:
            df['rho'] = df['rho'].fillna(100)
        return df

    df = convert_temperatures(df)
    df = fix_temperature_order(df)
    df = convert_rho(df)

    return df


def save_csv(records, filename):
    """Salva i rilievi in formato CSV"""
    if not records:
        print("Nessun dato da salvare.")
        return

    file_exists = os.path.exists(filename)
    fieldnames = list(records[0].keys())

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:  # scrive header solo la prima volta
            writer.writeheader()
        writer.writerows(records)


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


def main():

    print("📡 Importazione XML in corso...")
    xml_data = fetch_xml()

    if not xml_changed(xml_data):
        print("✅ Nessuna novità nei dati — stop.")
        return

    print("✅ Nuovi dati trovati, procedo...")

    # Parsing XML → dataframe iniziale
    rilievi = parse_xml(xml_data, CODICE_STAZIONE)
    rilievi_num = convert_all_rilievi(rilievi)

    # Override di test
    rilievi_num["TH10"][:] = 55
    rilievi_num["TH30"][:] = 54
    rilievi_num["PR"][:] = 0

    df = converti_aineva(rilievi_num)

    df['DataRilievo'] = pd.to_datetime(df['DataRilievo'])
    df = df.set_index('DataRilievo').sort_index()

    df = df.rename(columns={
        'Ta': 'TaG', 'Tmin': 'TminG', 'Tmax': 'TmaxG', 'HS': 'HSnum',
        'HN': 'HNnum', 'TH10': 'TH01G', 'TH30': 'TH03G'
    })

    df['Stagione'] = df.index.to_series().apply(calcola_stagione)

    # === FEATURE ENGINEERING ===
    df = calculate_day_of_season(df)
    df = calculate_snow_height_differences(df)
    df = calculate_new_snow(df)
    df = calculate_temperature(df)
    df = calculate_snow_temperature(df)
    df = calculate_swe(df)
    df = calculate_temperature_gradient(df)

    feature_set = [
        'HSnum', 'TH01G', 'PR', 'DayOfSeason', 'TmaxG_delta_5d', 'HS_delta_5d',
        'TH03G', 'HS_delta_1d', 'TmaxG_delta_3d', 'Precip_3d', 'TempGrad_HS',
        'HS_delta_2d', 'TmaxG_delta_2d', 'TminG_delta_5d', 'TminG_delta_3d',
        'Tsnow_delta_3d', 'TaG_delta_5d', 'Tsnow_delta_1d',
        'TmaxG_delta_1d', 'Precip_2d'
    ]

    df_selezionato = df[feature_set].copy()

    # === CHECK NaN ===
    last_row = df_selezionato.iloc[-1:]
    missing_features = last_row.columns[last_row.isna().any()]

    if len(missing_features) > 0:
        print("⚠️ Mancano dati in queste feature:")
        print(missing_features.tolist())
        print("⛔ Previsione NON possibile.")
        return

    print("✅ Nessun NaN nella riga più recente!")

    # === MODELLO AI ===
    model_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Valanghe\\PejoAvalancheML\\model')
    plots_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Valanghe\\PejoAvalancheML\\plots')

    svm_model = joblib.load(model_path / "svm_model.joblib")
    scaler = joblib.load(model_path / "scaler.joblib")
    X_background = joblib.load(model_path / "shap_background.joblib")

    explainer = shap.KernelExplainer(svm_model.predict_proba, X_background)

    X_scaled = scaler.transform(last_row)
    prob = svm_model.predict_proba(X_scaled)[0, 1]
    prediction = svm_model.predict(X_scaled)[0]

    # === SHAP Analysis ===
    shap_vals = explainer.shap_values(X_scaled)[0][:, 1]  # 20 valori
    shap_df = pd.DataFrame(shap_vals, index=last_row.columns, columns=["SHAP"])

    measured_values = last_row.iloc[0]
    df_comparativo = pd.concat([
        shap_df.T,
        measured_values.to_frame().T
    ])
    df_comparativo.index = ["SHAP_values", "Measured"]

    expected_value = explainer.expected_value[1]

    # === Force plot con unità e contributi ===
    units = {
        "HSnum": "cm", "TH01G": "°C", "PR": "mm", "DayOfSeason": "gg",
        "TmaxG_delta_5d": "°C/5d", "HS_delta_5d": "cm/5d", "TH03G": "°C",
        "HS_delta_1d": "cm/1d", "TmaxG_delta_3d": "°C/3d", "Precip_3d": "mm/3d",
        "TempGrad_HS": "°C/m", "HS_delta_2d": "cm/2d", "TmaxG_delta_2d": "°C/2d",
        "TminG_delta_5d": "°C/5d", "TminG_delta_3d": "°C/3d", "Tsnow_delta_3d": "°C/3d",
        "TaG_delta_5d": "°C/5d", "Tsnow_delta_1d": "°C/1d", "TmaxG_delta_1d": "°C/1d",
        "Precip_2d": "mm/2d"
    }

    labels = [
        f"{col} ({units[col]})" if col in units else col
        for col in last_row.columns
    ]

    feature_values_row = df_comparativo.loc["Measured"]

    labels_with_values = [
        f"{col} ({units.get(col, '')}) = {feature_values_row[col]:.2f}"
        for col in df_comparativo.columns
    ]

    # === RISULTATI ===

    if prediction == 1 and prob >= 0.6:
        print(f"🚨 **ALTO RISCHIO DI VALANGHE** (Probabilità: {prob:.3f})")
    elif prediction == 1 and prob >= 0.4:
        print(
            f"⚠️ **ATTENZIONE: Possibile valanga** (Probabilità: {prob:.3f})")
    else:
        print(f"✅ **Rischio valanghe basso** (Probabilità: {prob:.3f})")

    # Force plot
    shap.force_plot(
        expected_value,
        shap_vals,
        labels_with_values,
        contribution_threshold=0.15,
        matplotlib=True
    )

    print("✅ Analisi completata")


if __name__ == "__main__":
    main()
