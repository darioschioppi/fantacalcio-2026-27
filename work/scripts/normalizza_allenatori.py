#!/usr/bin/env python3
"""
Secondo passo di pulizia sui nomi allenatore raccolti da scrape_lineups.py.

Anche dopo il primo fix (usare mediaFirstName/mediaLastName invece di
shortName/displayName, che erano incoerenti tra provider), restano
duplicati residui perché ANCHE mediaFirstName/mediaLastName differiscono
leggermente tra provider per la stessa persona - tipicamente per il
middle name: "Eusebio Di Francesco" vs "Eusebio Luca Di Francesco",
"Cristian Chivu" vs "Cristian Eugen Chivu".

REGOLA DI FUSIONE: due nomi allenatore vengono considerati la STESSA
persona se e solo se:
  1. hanno lo stesso cognome (ultimo token del nome, normalizzato senza
     accenti/case), E
  2. il primo token del nome più corto coincide con il primo token del
     nome più lungo (stesso primo nome; il nome più lungo ha solo un
     secondo nome/middle name in più).
Questo evita di fondere per errore persone realmente diverse con lo
stesso cognome (es. "Filippo Inzaghi" e "Simone Inzaghi" sono due
allenatori distinti - primi nomi diversi, correttamente NON fusi).

Per ogni gruppo fuso si scegli come nome CANONICO il nome più lungo/
completo (più informazione, nessuna perdita).

Riscrive coach_name in:
  work/data/lineups_<stagione>.csv (tutti i file di stagione)
  work/data/lineups_storico_2015_2026.csv (aggregato)

Uso:
  python3 normalizza_allenatori.py
"""
import csv
import logging
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STORICO_PATH = DATA_DIR / "lineups_storico_2015_2026.csv"
LOG_PATH = DATA_DIR / "normalizza_allenatori_log.txt"

SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
           "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("normalizza_allenatori")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm_token(t):
    return strip_accents(t).lower()


def stesso_allenatore(nome_a, nome_b):
    tok_a = nome_a.split()
    tok_b = nome_b.split()
    if norm_token(tok_a[-1]) != norm_token(tok_b[-1]):
        return False
    return norm_token(tok_a[0]) == norm_token(tok_b[0])


def costruisci_mappa_canonica(tutti_i_nomi):
    """Raggruppa i nomi con union-find semplice basato su stesso_allenatore,
    poi scegli come rappresentante il nome più lungo di ogni gruppo."""
    nomi = sorted(set(tutti_i_nomi))
    gruppi = []  # lista di liste
    assegnato = {}

    for nome in nomi:
        trovato = False
        for gi, gruppo in enumerate(gruppi):
            if any(stesso_allenatore(nome, altro) for altro in gruppo):
                gruppo.append(nome)
                trovato = True
                break
        if not trovato:
            gruppi.append([nome])

    mappa = {}
    for gruppo in gruppi:
        canonico = max(gruppo, key=len)
        for nome in gruppo:
            mappa[nome] = canonico
        if len(gruppo) > 1:
            log.info("Fusi come '%s': %s", canonico, gruppo)

    return mappa


def riscrivi_csv(path, mappa):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_modificate = 0
    for row in rows:
        nome = row.get("coach_name")
        if nome and nome in mappa and mappa[nome] != nome:
            row["coach_name"] = mappa[nome]
            n_modificate += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return n_modificate


def main():
    tutti_i_nomi = []
    with open(STORICO_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("coach_name"):
                tutti_i_nomi.append(row["coach_name"])

    log.info("Nomi allenatore distinti prima della normalizzazione: %d", len(set(tutti_i_nomi)))
    mappa = costruisci_mappa_canonica(tutti_i_nomi)
    log.info("Nomi allenatore distinti dopo la normalizzazione: %d", len(set(mappa.values())))

    for stagione in SEASONS:
        path = DATA_DIR / f"lineups_{stagione}.csv"
        if path.exists():
            n = riscrivi_csv(path, mappa)
            log.info("%s: %d righe modificate", stagione, n)

    n_agg = riscrivi_csv(STORICO_PATH, mappa)
    log.info("Aggregato: %d righe modificate", n_agg)
    log.info("Normalizzazione completata.")


if __name__ == "__main__":
    main()
