# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:20:14 2025

@author: Christian
"""
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


def fetch_xml(URL):
    """Scarica il file XML"""
    r = requests.get(URL)
    r.raise_for_status()
    return r.content


def xml_changed(xml_data, HASH_FILE):
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
