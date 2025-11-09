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
