# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:16:27 2025

@author: Christian
"""
from telegrambot import send_telegram_message, send_telegram_image
from convert import convert_rilievo, convert_all_rilievi, converti_aineva
from getxml import fetch_xml, xml_changed, parse_xml
from transform import (calculate_day_of_season,
                       calculate_snow_height_differences,
                       calculate_new_snow, calculate_temperature,
                       calculate_swe, calculate_temperature_gradient,
                       calculate_snow_temperature, calcola_stagione)
from io import BytesIO
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd
import os

from pathlib import Path
import os

# # Imposta la cartella di lavoro sulla directory dello script
# try:
#     script_dir = Path(__file__).parent.resolve()
# except NameError:
#     script_dir = Path(os.getcwd()).resolve()

# os.chdir(script_dir)
# print(f"Directory corrente: {os.getcwd()}")

# ---user function----


# URL dei dati
URL = "http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve"
CODICE_STAZIONE = "21MB"
HASH_FILE = "last_hash.txt"   # qui salvo l’hash dell’ultimo XML scaricato

MODEL_PATH = Path(os.getenv("MODEL_PATH"))
PLOTS_PATH = Path(os.getenv("PLOTS_PATH"))


def main():
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    header = f"📊 *Previsione elaborata il {now}*"

    # print("📡 Importazione XML in corso...")
    xml_data = fetch_xml(URL)

    # if not xml_changed(xml_data):
    #     print("✅ Nessuna novità nei dati — stop.")
    #     return

    # print("✅ Nuovi dati trovati, procedo...")

    # Parsing XML → dataframe iniziale
    rilievi = parse_xml(xml_data, CODICE_STAZIONE)
    rilievi_num = convert_all_rilievi(rilievi)

    # Supponendo che rilievi_num abbia una colonna 'data'
    ultima_data = rilievi_num['DataRilievo'].max()

    # Se è stringa → converti in datetime
    if isinstance(ultima_data, str):
        ultima_data = datetime.fromisoformat(ultima_data)

    oggi = datetime.now()

    # Calcola differenza in giorni
    delta_giorni = (oggi.date() - ultima_data.date()).days

    # Se i dati non sono aggiornati
    if delta_giorni > 0:
        warning_msg = (
            f"⚠️ Attenzione: i dati non sono aggiornati!\n"
            f"L'ultimo rilievo è del {ultima_data.strftime('%d/%m/%Y')} "
            f"({delta_giorni} giorno{'i' if delta_giorni > 1 else ''} fa)."
        )

        # Invia messaggio Telegram
        send_telegram_message(warning_msg)

    # Override di test
    rilievi_num["TH10"][:] = 55
    rilievi_num["TH30"][:] = 54
    rilievi_num["PR"][:] = 70

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

    # print("✅ Nessun NaN nella riga più recente!")

    # === MODELLO AI ===

    svm_model = joblib.load(MODEL_PATH / "svm_model.joblib")
    scaler = joblib.load(MODEL_PATH / "scaler.joblib")
    X_background = joblib.load(MODEL_PATH / "shap_background.joblib")

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
        "HS_delta_1d": "cm/1d", "TmaxG_delta_3d": "°C/3d",
        "Precip_3d": "mm/3d", "TempGrad_HS": "°C/m", "HS_delta_2d": "cm/2d",
        "TmaxG_delta_2d": "°C/2d", "TminG_delta_5d": "°C/5d",
        "TminG_delta_3d": "°C/3d", "Tsnow_delta_3d": "°C/3d",
        "TaG_delta_5d": "°C/5d", "Tsnow_delta_1d": "°C/1d",
        "TmaxG_delta_1d": "°C/1d", "Precip_2d": "mm/2d"
    }

    feature_values_row = df_comparativo.loc["Measured"]

    labels_with_values = [
        f"{col} ({units.get(col, '')}) = {feature_values_row[col]:.2f}"
        for col in df_comparativo.columns
    ]

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
