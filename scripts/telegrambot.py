# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:23:40 2025

@author: Christian
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:16:27 2025

@author: Christian
"""

# ---user function----


# carica .env se presente (solo per sviluppo)


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
import transform
import getxml
import convert
import subprocess
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID devono essere impostati nelle env vars")


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


def handle_updates():
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    response = requests.get(url).json()
    return response["result"]


def main_bot():
    updates = handle_updates()
    for update in updates:
        if "message" in update:
            chat_id = update["message"]["chat"]["id"]
            text = update["message"].get("text", "")

            # Quando ricevi il comando /run
            if text.lower() == "/run":
                send_telegram_message("🟢 Avvio script main.py...")

                # Lancia lo script
                try:
                    subprocess.run(["python", "main.py"], check=True)
                    send_telegram_message("✅ Script completato!")
                except subprocess.CalledProcessError as e:
                    send_telegram_message(
                        f"❌ Errore durante l'esecuzione:\n{e}")


if __name__ == "__main__":
    main_bot()
