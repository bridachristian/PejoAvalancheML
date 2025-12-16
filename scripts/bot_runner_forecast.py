import os
import time
import requests
import subprocess
import traceback
from dotenv import load_dotenv

# ======================================================
# PATH DI LAVORO (come nel tuo script)
# ======================================================
os.chdir("C:/Users/Christian/OneDrive/Desktop/Valanghe/PejoAvalancheML/scripts")

# ======================================================
# ENV
# ======================================================
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # usato solo come fallback

if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN mancante")

# ======================================================
# TELEGRAM UTILS
# ======================================================


def send_telegram_message(message, chat_id=None):
    if chat_id is None:
        chat_id = CHAT_ID

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload, timeout=10)


def send_forecast_start_button(chat_id):
    """Invia un messaggio con un bottone inline per iniziare il forecast."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": "🤖 Premere il pulsante per inserire forecast:",
        "reply_markup": {
            "inline_keyboard": [[{"text": "🟢 Avvia Forecast", "callback_data": "start_forecast"}]]
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
        print("Errore API Telegram:", r)
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
     "🌙 Inserisci *nightclouds* (scrivi esattamente: sereno / poco nuvoloso / nuvoloso)"),
]

VALID_NIGHTCLOUDS = {"sereno", "poco nuvoloso", "nuvoloso"}


# ======================================================
# RUN MAIN_FORECAST
# ======================================================


def run_main_forecast(forecast_params, chat_id):
    try:
        send_telegram_message("🟢 Avvio analisi forecast...", chat_id)

        env = os.environ.copy()
        for k, v in forecast_params.items():
            env[k] = str(v)

        result = subprocess.run(
            ["python", "main_forecast.py"],
            env=env,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            send_telegram_message(
                f"📤 *Output modello:*\n```{result.stdout[:3500]}```",
                chat_id
            )

        if result.returncode == 0:
            send_telegram_message("✅ Analisi forecast completata!", chat_id)
        else:
            send_telegram_message(
                f"❌ *Errore modello:*\n```{result.stderr[:3500]}```",
                chat_id
            )

    except Exception:
        tb = traceback.format_exc()
        send_telegram_message(
            f"❌ *Eccezione critica:*\n```{tb[:3500]}```",
            chat_id
        )

# ======================================================
# MAIN LOOP
# ======================================================


def main():
    last_update_id = None
    send_forecast_start_button(CHAT_ID)

    while True:
        updates = handle_updates(
            offset=(last_update_id + 1) if last_update_id else None)
        for update in updates:
            last_update_id = update["update_id"]

            # -----------------
            # Bottone premuto
            # -----------------
            if "callback_query" in update:
                callback = update["callback_query"]
                chat_id = callback["message"]["chat"]["id"]
                data = callback["data"]
                if data == "start_forecast":
                    user_state[chat_id] = {"step": 0, "values": {}}
                    send_telegram_message(PARAM_FLOW[0][2], chat_id)
                continue

            if "message" not in update:
                continue

            msg = update["message"]
            chat_id = msg["chat"]["id"]
            text = msg.get("text", "").strip().lower()

            # Comando di annullamento
            if text == "/cancel":
                user_state.pop(chat_id, None)
                send_telegram_message("❌ Operazione annullata.", chat_id)
                continue

            # -------------------------
            # DIALOGO FORECAST
            # -------------------------
            if chat_id in user_state:
                state = user_state[chat_id]
                step = state["step"]
                key, cast, prompt = PARAM_FLOW[step]

                try:
                    value = cast(text)

                    # === Validazioni incrociate ===
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

                    # === Validazione nightclouds_forecast ===
                    if key == "nightclouds_forecast":
                        value = value.replace("_", " ").lower()
                        if value not in VALID_NIGHTCLOUDS:
                            send_telegram_message(
                                "⚠️ Valore non valido. Scrivi: sereno / poco nuvoloso / nuvoloso", chat_id)
                            continue

                    # Salva valore
                    state["values"][key] = value
                    state["step"] += 1

                    # Passa al prossimo step o esegue main_forecast
                    if state["step"] < len(PARAM_FLOW):
                        send_telegram_message(
                            PARAM_FLOW[state["step"]][2], chat_id)
                    else:
                        # Riepilogo
                        summary = "*📋 Parametri forecast inseriti:*\n"
                        for k, v in state["values"].items():
                            summary += f"- *{k}*: {v}\n"
                        send_telegram_message(summary, chat_id)
                        run_main_forecast(state["values"], chat_id)
                        del user_state[chat_id]

                except ValueError:
                    send_telegram_message(
                        "⚠️ Valore non valido, riprova.", chat_id)

        time.sleep(2)


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    main()
