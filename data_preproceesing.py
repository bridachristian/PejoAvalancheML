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
from transform import calculate_day_of_season

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


TOKEN = "8049162233:AAGetIIn76Msresu39P6WhE3RsQWd4Oms2M"
CHAT_ID = "467116928"


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

# Invia via Telegram


def send_telegram_image(image_bytes):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    files = {"photo": image_bytes}
    data = {"chat_id": CHAT_ID}
    requests.post(url, files=files, data=data)


# TOKEN = "8049162233:AAGetIIn76Msresu39P6WhE3RsQWd4Oms2M"
# response = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates").json()
# print(response)


def main():
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    header = f"📊 *Previsione elaborata il {now}*"

    print("📡 Importazione XML in corso...")
    xml_data = fetch_xml()

    # if not xml_changed(xml_data):
    #     print("✅ Nessuna novità nei dati — stop.")
    #     return

    # print("✅ Nuovi dati trovati, procedo...")

    # Parsing XML → dataframe iniziale
    rilievi = parse_xml(xml_data, CODICE_STAZIONE)
    rilievi_num = convert_all_rilievi(rilievi)

    # Override di test
    rilievi_num["TH10"][:] = 55
    rilievi_num["TH30"][:] = np.nan
    rilievi_num["PR"][:] = np.nan

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
        error_msg = f"⚠️ *Attenzione! Mancano dati in queste feature:*\n{
            ', '.join(missing_features)}\n⛔ Previsione NON possibile."
        final_message = f"{header}\n{error_msg}"

        send_telegram_message(final_message)
        # print(missing_features.tolist())
        # print("⛔ Previsione NON possibile.")
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

    # if prediction == 1 and prob >= 0.6:
    #     print(f"🚨 **ALTO RISCHIO DI VALANGHE** (Probabilità: {prob:.3f})")
    # elif prediction == 1 and prob >= 0.4:
    #     print(
    #         f"⚠️ **ATTENZIONE: Possibile valanga** (Probabilità: {prob:.3f})")
    # else:
    #     print(f"✅ **Rischio valanghe basso** (Probabilità: {prob:.3f})")

    # Force plot
    shap.force_plot(
        expected_value,
        shap_vals,
        labels_with_values,
        contribution_threshold=0.15,
        matplotlib=True
    )

    print("✅ Analisi completata")

    # === INVIO SU TELEGRAM ===

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    header = f"📊 *Previsione elaborata il {now}*"

    if prediction == 1 and prob >= 0.6:
        risk_msg = f"🚨 *ALTO RISCHIO DI VALANGHE* (Probabilità: {
            100*prob:.0f}%)"
    elif prediction == 1 and prob >= 0.4:
        risk_msg = f"⚠️ *ATTENZIONE: Possibile valanga* (Probabilità: {
            100*prob:.0f}%)"
    else:
        risk_msg = f"✅ *Rischio valanghe basso* (Probabilità: {100*prob:.0f}%)"

    # Messaggio finale unico
    final_message = f"{header}\n{risk_msg}"

    # Invia il messaggio unico
    send_telegram_message(final_message)

    # === Genera il grafico SHAP ===
    shap.plots.force(
        shap_values=shap_vals,
        base_value=expected_value,
        feature_names=labels_with_values,
        contribution_threshold=0.15,
        text_rotation=90,
        matplotlib=True,
        show=False
    )

    # Salva il plot su buffer e invia via Telegram
    buf = BytesIO()
    plt.gcf().savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    send_telegram_image(buf)
    plt.close()


if __name__ == "__main__":
    main()
