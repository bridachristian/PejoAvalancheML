# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:34:43 2025

@author: Christian
"""

import requests
import subprocess
import time
import traceback

TOKEN = "8049162233:AAGetIIn76Msresu39P6WhE3RsQWd4Oms2M"
CHAT_ID = "467116928"


def send_telegram_message(message):
    """Invia un messaggio testuale a Telegram."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)


def send_telegram_image(image_bytes):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    files = {"photo": image_bytes}
    data = {"chat_id": CHAT_ID}
    requests.post(url, files=files, data=data)


def handle_updates(offset=None):
    """Prende gli ultimi messaggi dal bot."""
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 10}
    response = requests.get(url, params=params).json()
    return response["result"]


def run_main_script():
    """Esegue main.py e cattura errori."""
    try:
        send_telegram_message("🟢 Avvio script main.py...")
        # Lancia lo script; cattura output e errori
        result = subprocess.run(
            ["python", "main.py"], capture_output=True, text=True
        )
        if result.returncode == 0:
            send_telegram_message("✅ Script completato correttamente!")
        else:
            # Se c'è un errore, invia stdout e stderr
            msg = f"❌ Errore nello script:\nSTDOUT:\n{
                result.stdout}\nSTDERR:\n{result.stderr}"
            send_telegram_message(msg)
    except Exception as e:
        tb = traceback.format_exc()
        send_telegram_message(f"❌ Eccezione durante l'esecuzione:\n{tb}")


def main():
    last_update_id = None
    while True:
        updates = handle_updates(
            offset=(last_update_id + 1) if last_update_id else None)
        for update in updates:
            last_update_id = update["update_id"]

            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                text = update["message"].get("text", "")

                # Comando per lanciare main.py
                if text.lower() == "/run":
                    run_main_script()
        time.sleep(3)  # controlla ogni 3 secondi


if __name__ == "__main__":
    send_telegram_message("🤖 Bot avviato e pronto.")
    main()
