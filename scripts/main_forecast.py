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
import numpy as np

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
HASH_FILE = "last_hash.txt"   # qui salvo l’hash dell’ultimo XML scaricato

CODICE_STAZIONE = os.getenv("CODICE_STAZIONE")
MODEL_PATH = Path(os.getenv("MODEL_PATH"))
PLOTS_PATH = Path(os.getenv("PLOTS_PATH"))

Ta_forecast = float(os.getenv("Ta_forecast", -1))
Tmin_forecast = float(os.getenv("Tmin_forecast", -5))
Tmax_forecast = float(os.getenv("Tmax_forecast", 5))
HN_forecast = float(os.getenv("HN_forecast", 10))
nightclouds_forecast = os.getenv("nightclouds_forecast", "sereno")


def estimate_density(Ta):
    """
    Ritorna densità stimata della neve fresca in kg/m³.
    Modello semplificato basato su temperatura.
    """
    if Ta <= -8:
        return 50
    elif Ta <= -4:
        return 70
    elif Ta <= -1:
        return 90
    elif Ta <= 1:
        return 110
    elif Ta <= 3:
        return 140
    else:
        return 170


def estimate_snow_temperatures(HS, nightclouds_forecast, TH10_last=None, TH30_last=None):
    """
    Stima le temperature della neve a 10 e 30 cm dalla superficie.
    - Se TH10_last e TH30_last sono disponibili → applica delta notturni
    - Altrimenti crea un profilo lineare tra suolo (0°C) e superficie

    HS = altezza neve totale (cm)
    nightclouds_forecast = 'sereno', 'poco_nuvoloso', 'nuvoloso'
    """
    # Definiamo i delta notturni (raffreddamento verso la superficie)
    if nightclouds_forecast == 'sereno':
        delta_TH10 = -4
        delta_TH30 = -2
    elif nightclouds_forecast == 'poco_nuvoloso':
        delta_TH10 = -2
        delta_TH30 = -1
    else:  # nuvoloso
        delta_TH10 = -0.5
        delta_TH30 = -0.2

    TH10 = TH10_last + delta_TH10

    # Se abbiamo dati precedenti, applico delta
    if TH30_last is not None and not np.isnan(TH30_last):
        # abbiamo dato storico → applica delta
        TH30 = TH30_last + delta_TH30
    else:
        # Se prima era NaN ma ora HS >= 30 → calcolo TH30 come interpolazione tra TH10 e suolo
        if HS >= 30:
            # distanza tra TH10 (10 cm sotto superficie) e suolo (0°C)
            # proporzione lineare: TH30 a 30 cm sotto superficie
            # TH10 si trova a HS - 10 cm dal suolo
            TH30 = TH10 * ((HS - 30) / (HS - 10))
        else:
            # ancora troppo poca neve → non c'è TH30
            TH30 = 0

    return TH10, TH30


def create_forecast_row(rilievi_num):
    # prendi l’ultima riga
    last = rilievi_num.iloc[0].copy()

    # nuova data = ultima + 1 giorno
    new_date = pd.to_datetime(last["DataRilievo"]) + pd.Timedelta(days=1)

    # --- Temperature dell'aria ---
    last["Ta"] = Ta_forecast
    last["Tmin"] = Tmin_forecast
    last["Tmax"] = Tmax_forecast

    # --- HN e HS ---
    last_HS = last["HS"]
    new_HS = last_HS + HN_forecast
    last["HN"] = HN_forecast
    last["HS"] = new_HS

    # --- Densità neve ---
    density = estimate_density(Ta_forecast)
    last["rho"] = density

    # --- PR (penetrazione) ---
    if HN_forecast > 0:
        last["PR"] = last["PR"] + HN_forecast
    # se HN = 0 mantieni la precedente

    # --- Temperature neve (TH10 / TH30) ---
    TH10, TH30 = estimate_snow_temperatures(HS=last['HS'],
                                            TH10_last=last['TH10'],
                                            TH30_last=last['TH30'],
                                            nightclouds_forecast=nightclouds_forecast
                                            )
    last["TH10"] = TH10
    last["TH30"] = TH30

    # nuova data
    last["DataRilievo"] = new_date

    return last


def plot_shap_oggi_domani_single(shap_df, top_n=20):
    SHAP_NEG = "#258ae5"   # SHAP blue
    SHAP_POS = "#ff0e57"   # SHAP magenta

    # Seleziona feature più importanti
    df = shap_df[["Oggi", "Domani"]].copy()
    df["max_abs"] = df.abs().max(axis=1)
    df = df.sort_values("max_abs", ascending=False).head(top_n)
    df = df.drop(columns="max_abs")

    # Ordine per Oggi
    df = df.sort_values("Oggi")

    y = np.arange(len(df))
    h = 0.35  # spessore barre

    plt.figure(figsize=(8, 6))

    # OGGI
    colors_oggi = df["Oggi"].apply(lambda x: SHAP_POS if x > 0 else SHAP_NEG)
    plt.barh(y + h/2, df["Oggi"], height=h, color=colors_oggi, label="Oggi")

    # DOMANI
    colors_domani = df["Domani"].apply(
        lambda x: SHAP_POS if x > 0 else SHAP_NEG)
    plt.barh(y - h/2, df["Domani"], height=h,
             color=colors_domani, alpha=0.5, label="Domani")

    # Asse Y
    plt.yticks(y, df.index)

    # Zero line
    plt.axvline(0, color="black", linewidth=1)

    plt.xlabel("SHAP value")
    plt.title("Contributo delle feature al rischio valanghe (OGGI vs DOMANI)")
    plt.legend()
    plt.tight_layout()


def main():
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    header = f"📊 *Previsione elaborata il {now_str}*"

    # --- Importazione XML ---
    xml_data = fetch_xml(URL)
    rilievi = parse_xml(xml_data, CODICE_STAZIONE)
    rilievi_num = convert_all_rilievi(rilievi)

    # --- CREA RIGA DI PREVISIONE ---
    forecast_row = create_forecast_row(rilievi_num)
    rilievi_num = pd.concat(
        [pd.DataFrame([forecast_row]), rilievi_num],
        ignore_index=True
    )

    # --- Controllo aggiornamento dati ---
    ultima_data = rilievi_num['DataRilievo'].max()
    if isinstance(ultima_data, str):
        ultima_data = datetime.fromisoformat(ultima_data)
    delta_giorni = (datetime.now().date() - ultima_data.date()).days
    if delta_giorni > 0:
        send_telegram_message(
            f"⚠️ Attenzione: i dati non sono aggiornati!\n"
            f"L'ultimo rilievo è del {ultima_data.strftime('%d/%m/%Y')} "
            f"({delta_giorni} giorno{'i' if delta_giorni > 1 else ''} fa)."
        )

    # --- Conversione a formato AI-Neva ---
    df = converti_aineva(rilievi_num)
    df['DataRilievo'] = pd.to_datetime(df['DataRilievo'])
    df = df.set_index('DataRilievo').sort_index()
    df = df.rename(columns={
        'Ta': 'TaG', 'Tmin': 'TminG', 'Tmax': 'TmaxG',
        'HS': 'HSnum', 'HN': 'HNnum', 'TH10': 'TH01G', 'TH30': 'TH03G'
    })
    df['Stagione'] = df.index.to_series().apply(calcola_stagione)

    # --- FEATURE ENGINEERING ---
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
    last_two_rows = df_selezionato.iloc[-2:]

    # forecast_row = last_two_rows.iloc[1]  # riga di previsione (domani)
    # today_row = last_two_rows.iloc[0]     # riga di oggi

    # Etichette per Telegram
    labels = ["Oggi", "Domani"]

    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    header = f"📊 *Previsione elaborata il {now}*"
    # Messaggio finale unico
    final_message = f"{header}"

    # Invia il messaggio unico
    send_telegram_message(final_message)

    shap_df = pd.DataFrame(index=last_two_rows.columns)

    for i, label in enumerate(labels):
        # print(i)
        # print(label)
        # Righe come Serie
        last_row = last_two_rows.iloc[i:i+1]
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

        # shap_df = pd.DataFrame(
        #     shap_vals, index=last_row.columns, columns=["SHAP"])

        shap_df[label] = shap_vals

        # measured_values = last_row.iloc[0]
        # df_comparativo = pd.concat([
        #     shap_df.T,
        #     measured_values.to_frame().T
        # ])
        # df_comparativo.index = ["SHAP_values", "Measured"]

        measured_values = last_row.iloc[0]
        # prendi solo la prima riga dei valori SHAP
        shap_row = shap_df.T.iloc[0]

        df_comparativo = pd.DataFrame([shap_row, measured_values],
                                      index=["SHAP_values", "Measured"])

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

        row_date = last_row.index[0].strftime("%d/%m/%Y")
        header_inside = f"*Valida per: {label} - {row_date}*"

        if prob >= 0.6:
            risk_msg = f"🚨 *ALTO RISCHIO DI VALANGHE* (Probabilità: {
                100*prob:.0f}%)"
        elif prob >= 0.4:
            risk_msg = f"⚠️ *ATTENZIONE: Possibile valanga* (Probabilità: {
                100*prob:.0f}%)"
        else:
            risk_msg = f"✅ *Rischio valanghe basso* (Probabilità: {
                100*prob:.0f}%)"

        # Messaggio finale unico
        final_message = f"{header_inside}\n{risk_msg}"

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

    shap_df['Differenza'] = shap_df['Domani'] - shap_df['Oggi']

    plot_shap_oggi_domani_single(shap_df)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    send_telegram_image(buf)
    plt.close()


if __name__ == "__main__":
    main()
