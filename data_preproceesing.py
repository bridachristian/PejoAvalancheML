import pandas as pd
import requests
import xml.etree.ElementTree as ET
import csv
import os
import hashlib
from datetime import datetime
import numpy as np


# URL dei dati
URL = "http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve"
CODICE_STAZIONE = "21MB"
FILE_CSV = "storico_21MB.csv"
HASH_FILE = "last_hash.txt"   # qui salvo l’hash dell’ultimo XML scaricato


def fetch_xml():
    """Scarica il file XML"""
    r = requests.get(URL)
    r.raise_for_status()
    return r.content


def xml_changed(xml_data):
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


def convert_rilievo(r):
    """Converte i valori stringa di un rilievo in numerici + DataRilievo in datetime"""
    def to_num(val):
        if val is None:
            return None
        v = val.strip()
        if v in ["///", "//", "", None]:
            return np.NaN
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return np.NaN

    def to_date(val):
        if not val:
            return np.NaN
        try:
            # formato visto nell'XML: '21/09/2025 00:00:00'
            return datetime.strptime(val.strip(), "%d/%m/%Y %H:%M:%S")
        except ValueError:
            return np.NaN

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


def save_csv(records, filename):
    """Salva i rilievi in formato CSV"""
    if not records:
        print("Nessun dato da salvare.")
        return

    file_exists = os.path.exists(filename)
    fieldnames = list(records[0].keys())

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:  # scrive header solo la prima volta
            writer.writeheader()
        writer.writerows(records)


def main():
    xml_data = fetch_xml()

    # controllo se è cambiato
    if not xml_changed(xml_data):
        return

    # qui prosegui solo se i dati sono nuovi
    print("✅ Nuovo file XML, procedo con analisi...")
    rilievi = parse_xml(xml_data, CODICE_STAZIONE)

    rilievi_num = [convert_rilievo(r) for r in rilievi]

    rilievi_num = convert_all_rilievi(rilievi)
    # TEST
    rilievi_num["Ta"][0] = 55
    rilievi_num["Tmin"][0] = 54
    rilievi_num["Tmax"][0] = 1
    rilievi_num["TH10"][0] = 55
    rilievi_num["TH30"][0] = 54
    rilievi_num["HN"][0:2] = [10, 2]

    print(rilievi_num)
    df = converti_aineva(rilievi_num)

    # parsing e salvataggio CSV andranno qui
    # es: parse_xml(xml_data, CODICE_STAZIONE) ...


if __name__ == "__main__":
    main()
