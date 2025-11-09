# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:22:27 2025

@author: Christian
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:16:27 2025

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
import getxml
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
