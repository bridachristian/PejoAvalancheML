import os
import time
import requests
import subprocess
from dotenv import load_dotenv

# ======================================================
# PATH DI LAVORO
# ======================================================
os.chdir("C:/Users/Christian/OneDrive/Desktop/Valanghe/PejoAvalancheML/scripts")

# ======================================================
# ENV
# ======================================================
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # fallback

if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN mancante")

# ======================================================
# UTILS TELEGRAM
# ======================================================


def send_telegram_message(message, chat_id=None):
    if chat_id is None:
        chat_id = CHAT_ID
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload, timeout=10)


def send_telegram_image(image_bytes, chat_id=None):
    if chat_id is None:
        chat_id = CHAT_ID
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    files = {"photo": image_bytes}
    data = {"chat_id": chat_id}
    requests.post(url, files=files, data=data, timeout=20)


def send_forecast_buttons(chat_id):
    """Invia bottoni: OGGI o FULL forecast"""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": "🤖 Scegli il tipo di previsione:",
        "reply_markup": {
            "inline_keyboard": [
                [
                    {"text": "📅 Previsione OGGI", "callback_data": "forecast_today"},
                    {"text": "🟢 Forecast completo",
                        "callback_data": "start_forecast"}
                ]
            ]
        },
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload, timeout=10)


def handle_updates(offset=None):
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    params = {"timeout": 15}
    if offset:
        params["offset"] = offset
    try:
        r = requests.get(url, params=params, timeout=20).json()
    except Exception as e:
        print("Errore getUpdates:", e)
        return []
    if not r.get("ok", False):
        return []
    return r.get("result", [])


# ======================================================
# STATO DIALOGO
# ======================================================
user_state = {}  # chat_id -> {"step": int, "values": dict}

PARAM_FLOW = [
    ("Ta_forecast", float, "🌡️ Inserisci *Ta_forecast* (°C, es: 0)"),
    ("Tmin_forecast", float, "🌡️ Inserisci *Tmin_forecast* (°C, es: -3)"),
    ("Tmax_forecast", float, "🌡️ Inserisci *Tmax_forecast* (°C, es: 7)"),
    ("HN_forecast", float, "❄️ Inserisci *HN_forecast* (cm, es: 20)"),
    ("nightclouds_forecast", str,
     "🌙 Inserisci *nightclouds* (sereno / poco nuvoloso / nuvoloso)"),
]

VALID_NIGHTCLOUDS = {"sereno", "poco nuvoloso", "nuvoloso"}


# ======================================================
# RUN MAIN_FORECAST
# ======================================================
def run_main_forecast(forecast_params, chat_id):
    """Avvia main_forecast_test.py in background"""
    try:
        send_telegram_message("🟢 Avvio analisi forecast...", chat_id)

        env = os.environ.copy()
        for k, v in forecast_params.items():
            env[k] = str(v)

        subprocess.Popen(
            ["python", "main_forecast_2buttons.py", "full"],
            env=env
        )

        # send_telegram_message(
        #     "✅ Analisi forecast avviata in background.", chat_id)

    except Exception as e:
        send_telegram_message(f"❌ Errore nell'avvio forecast: {e}", chat_id)


def run_today_forecast(chat_id):
    """Avvia previsione per oggi in background"""
    try:
        send_telegram_message("🟢 Avvio previsione OGGI...", chat_id)

        env = os.environ.copy()
        subprocess.Popen(
            ["python", "main_forecast_2buttons.py", "today"],
            env=env
        )

        # send_telegram_message(
        #     "✅ Previsione OGGI avviata in background.", chat_id)

    except Exception as e:
        send_telegram_message(
            f"❌ Errore nell'avvio previsione oggi: {e}", chat_id)


# ======================================================
# MAIN LOOP
# ======================================================
def main():
    last_update_id = None
    send_forecast_buttons(CHAT_ID)

    while True:
        updates = handle_updates(
            offset=(last_update_id + 1) if last_update_id else None)
        for update in updates:
            last_update_id = update["update_id"]

            # ---------- CALLBACK QUERY ----------
            if "callback_query" in update:
                callback = update["callback_query"]
                chat_id = callback["message"]["chat"]["id"]
                data = callback["data"]

                # Rispondi subito al callback per sbloccare i bottoni
                url = f"https://api.telegram.org/bot{
                    TOKEN}/answerCallbackQuery"
                requests.post(url, data={"callback_query_id": callback["id"]})

                if data == "start_forecast":
                    user_state[chat_id] = {"step": 0, "values": {}}
                    send_telegram_message(PARAM_FLOW[0][2], chat_id)

                elif data == "forecast_today":
                    run_today_forecast(chat_id)

                continue

            # ---------- MESSAGGI ----------
            if "message" not in update:
                continue
            msg = update["message"]
            chat_id = msg["chat"]["id"]
            text = msg.get("text", "").strip()

            # /cancel
            if text.lower() == "/cancel":
                user_state.pop(chat_id, None)
                send_telegram_message("❌ Operazione annullata.", chat_id)
                continue

            # ---------- DIALOGO PARAMETRI ----------
            if chat_id in user_state:
                state = user_state[chat_id]
                step = state["step"]
                key, cast, prompt = PARAM_FLOW[step]

                try:
                    value = cast(text)
                    # Validazioni incrociate
                    Tmin = state["values"].get("Tmin_forecast")
                    Tmax = state["values"].get("Tmax_forecast")
                    Ta = state["values"].get("Ta_forecast")

                    if key == "Ta_forecast":
                        if Tmin is not None and value < Tmin:
                            send_telegram_message(
                                f"⚠️ Ta_forecast ({value}) < Tmin_forecast ({Tmin}). Riprova.", chat_id)
                            continue
                        if Tmax is not None and value > Tmax:
                            send_telegram_message(
                                f"⚠️ Ta_forecast ({value}) > Tmax_forecast ({Tmax}). Riprova.", chat_id)
                            continue

                    if key == "Tmin_forecast":
                        if Tmax is not None and value > Tmax:
                            send_telegram_message(
                                f"⚠️ Tmin_forecast ({value}) > Tmax_forecast ({Tmax}). Riprova.", chat_id)
                            continue
                        if Ta is not None and value > Ta:
                            send_telegram_message(
                                f"⚠️ Tmin_forecast ({value}) > Ta_forecast ({Ta}). Riprova.", chat_id)
                            continue

                    if key == "Tmax_forecast":
                        if Tmin is not None and value < Tmin:
                            send_telegram_message(
                                f"⚠️ Tmax_forecast ({value}) < Tmin_forecast ({Tmin}). Riprova.", chat_id)
                            continue
                        if Ta is not None and value < Ta:
                            send_telegram_message(
                                f"⚠️ Tmax_forecast ({value}) < Ta_forecast ({Ta}). Riprova.", chat_id)
                            continue

                    if key == "nightclouds_forecast":
                        value = value.replace("_", " ").lower()
                        if value not in VALID_NIGHTCLOUDS:
                            send_telegram_message(
                                "⚠️ Valore non valido. Scrivi: sereno / poco nuvoloso / nuvoloso", chat_id)
                            continue

                    # salva valore
                    state["values"][key] = value
                    state["step"] += 1

                    # prossimo step o esegui forecast
                    if state["step"] < len(PARAM_FLOW):
                        send_telegram_message(
                            PARAM_FLOW[state["step"]][2], chat_id)
                    else:
                        summary = "*📋 Parametri forecast inseriti:*\n"
                        for k, v in state["values"].items():
                            summary += f"- *{k}*: {v}\n"
                        send_telegram_message(summary, chat_id)
                        run_main_forecast(state["values"], chat_id)
                        del user_state[chat_id]

                except ValueError:
                    send_telegram_message(
                        "⚠️ Valore non valido, riprova.", chat_id)

        time.sleep(1)


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    main()
