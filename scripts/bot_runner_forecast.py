# -*- coding: utf-8 -*-
"""
bot_runner_forecast.py
Bot Telegram per inserimento interattivo dei parametri e lancio di main_forecast.py
"""

import os
from dotenv import load_dotenv
import requests
import subprocess
import time
import traceback

# Carica variabili ambiente da .env (solo in sviluppo)
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID devono essere impostati nelle env vars"
    )

# Stato utente: memorizza i parametri inseriti durante il flusso
user_inputs = {}  # es. {chat_id: {"Ta_forecast": 0, ...}}


def send_telegram_message(message):
    """Invia un messaggio testuale a Telegram."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)


def send_telegram_image(image_bytes):
    """Invia un'immagine (bytes) a Telegram."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    files = {"photo": image_bytes}
    data = {"chat_id": CHAT_ID}
    requests.post(url, files=files, data=data)


def handle_updates(offset=None):
    """Recupera aggiornamenti dal bot."""
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 10}
    try:
        response = requests.get(url, params=params).json()
    except Exception as e:
        print("Errore getUpdates:", e)
        return []

    if not response.get("ok", False):
        print("Errore API Telegram:", response.get("description", response))
        return []

    return response.get("result", [])


def run_main_script_with_params(inputs):
    """Avvia main_forecast.py passando i parametri come variabili d'ambiente."""
    env = os.environ.copy()
    for k, v in inputs.items():
        env[k] = str(v)
    subprocess.run(["python", "main_forecast.py"], env=env)


def main():
    last_update_id = None
    send_telegram_message(
        "🤖 Bot avviato e pronto. Digita /run per inserire i parametri di forecast.")

    while True:
        updates = handle_updates(
            offset=(last_update_id + 1) if last_update_id else None)
        for update in updates:
            last_update_id = update["update_id"]

            if "message" not in update:
                continue

            chat_id = update["message"]["chat"]["id"]
            text = update["message"].get("text", "").strip()

            # Comando iniziale
            if text.lower() == "/run":
                user_inputs[chat_id] = {}
                send_telegram_message(
                    "📊 Inserisci Ta_forecast (temperatura aria prevista, es: 0):")
                continue

            # Flusso interattivo di inserimento
            if chat_id in user_inputs:
                inputs = user_inputs[chat_id]

                try:
                    if "Ta_forecast" not in inputs:
                        inputs["Ta_forecast"] = float(text)
                        send_telegram_message(
                            "📊 Inserisci Tmin_forecast (es: -3):")
                    elif "Tmin_forecast" not in inputs:
                        inputs["Tmin_forecast"] = float(text)
                        send_telegram_message(
                            "📊 Inserisci Tmax_forecast (es: 7):")
                    elif "Tmax_forecast" not in inputs:
                        inputs["Tmax_forecast"] = float(text)
                        send_telegram_message(
                            "📊 Inserisci HN_forecast (nuova neve in cm, es: 20):")
                    elif "HN_forecast" not in inputs:
                        inputs["HN_forecast"] = float(text)
                        send_telegram_message(
                            "📊 Inserisci nightclouds (sereno/poco_nuvoloso/nuvoloso):")
                    elif "nightclouds" not in inputs:
                        inputs["nightclouds"] = text.lower()
                        send_telegram_message(
                            "✅ Tutti i parametri inseriti! Avvio analisi...")

                        # Avvia lo script con i parametri
                        run_main_script_with_params(inputs)

                        # Pulisci lo stato utente
                        del user_inputs[chat_id]
                except ValueError:
                    send_telegram_message(
                        "⚠️ Valore non valido. Riprova con un numero o testo corretto.")

        time.sleep(1)  # controlla aggiornamenti ogni 2 secondi


if __name__ == "__main__":
    main()
