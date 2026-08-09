#!/usr/bin/env python3
"""
Estende feature_dataset_v1.csv aggiungendo:
  - modulo/formazione tattica (squadra e avversario) da lineups_storico_2015_2026.csv
  - allenatore (squadra e avversario) dalla stessa fonte - copertura limitata:
    presente solo dal 2023-24 in avanti (0% prima, vedi scrape_lineups.py)
  - classifica dinamica PRE-partita (punti/diff reti/posizione, squadra e
    avversario) da classifica_dinamica_storico_2015_2026.csv - calcolata
    SENZA leakage (solo partite giocate prima della giornata in questione)

Join lineups: i match_id di Lega Serie A (hash Deltatre) sono diversi dai
match_id di fantacalcio.it, quindi il join usa (stagione, data, squadra)
con normalizzazione nomi squadra (stessa mappa già usata per il contesto
squadra Lega Serie A in build_feature_dataset.py: Chievo->Chievoverona,
SPAL->Spal) e tolleranza ±1 giorno per lo scarto UTC/locale (stesso
problema già riscontrato e risolto nel matching Understat).

Join classifica dinamica: diretto su (stagione, giornata, squadra) - nessuna
normalizzazione nomi necessaria perché entrambe le fonti derivano dallo
stesso file voti_storici_2015_2026.csv.

Output: work/data/feature_dataset_v2.csv
"""
import csv
import logging
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
V1_PATH = DATA_DIR / "feature_dataset_v1.csv"
LINEUPS_PATH = DATA_DIR / "lineups_storico_2015_2026.csv"
CLASSIFICA_PATH = DATA_DIR / "classifica_dinamica_storico_2015_2026.csv"
OUT_PATH = DATA_DIR / "feature_dataset_v2.csv"
LOG_PATH = DATA_DIR / "build_feature_dataset_v2_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("build_features_v2")

TEAM_NAME_MAP_LINEUPS = {
    "Chievo": "Chievoverona",
    "SPAL": "Spal",
}


def norm_team_lineups(name):
    return TEAM_NAME_MAP_LINEUPS.get(name, name)


def carica_lineups():
    """dict {(stagione, data, team_title): row}"""
    lineups = {}
    with open(LINEUPS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = (row["match_date"] or "")[:10]
            key = (row["stagione"], data, row["team_title"])
            lineups[key] = row
    log.info("Caricate %d righe lineups", len(lineups))
    return lineups


def trova_lineup(lineups, stagione, data_str, team_title):
    key = (stagione, data_str, team_title)
    if key in lineups:
        return lineups[key]
    try:
        y, m, d = (int(x) for x in data_str.split("-"))
        base = date(y, m, d)
    except (ValueError, AttributeError):
        return None
    for delta in (1, -1):
        alt_data = (base + timedelta(days=delta)).isoformat()
        alt_key = (stagione, alt_data, team_title)
        if alt_key in lineups:
            return lineups[alt_key]
    return None


def carica_classifica():
    """dict {(stagione, giornata, squadra): row}"""
    classifica = {}
    with open(CLASSIFICA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["stagione"], row["giornata"], row["squadra"])
            classifica[key] = row
    log.info("Caricate %d righe classifica dinamica", len(classifica))
    return classifica


def main():
    lineups = carica_lineups()
    classifica = carica_classifica()

    n_rows = 0
    n_match_modulo_squadra = 0
    n_match_modulo_avversario = 0
    n_match_allenatore_squadra = 0
    n_match_allenatore_avversario = 0
    n_match_classifica_squadra = 0
    n_match_classifica_avversario = 0

    nuove_colonne = [
        "squadra_modulo", "avversario_modulo",
        "squadra_allenatore", "avversario_allenatore",
        "squadra_posizione_pre", "squadra_punti_pre", "squadra_diff_reti_pre",
        "avversario_posizione_pre", "avversario_punti_pre", "avversario_diff_reti_pre",
    ]

    with open(V1_PATH, newline="", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + nuove_colonne
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            n_rows += 1
            stagione = row["stagione"]
            giornata = row["giornata"]
            data = row["data"]
            squadra = row["squadra_giocatore"]
            squadra_casa = row["squadra_casa"]
            squadra_ospite = row["squadra_ospite"]
            avversario = squadra_ospite if squadra == squadra_casa else squadra_casa

            out_row = dict(row)

            # --- modulo/allenatore squadra ---
            team_lu = norm_team_lineups(squadra)
            lu_squadra = trova_lineup(lineups, stagione, data, team_lu)
            if lu_squadra:
                out_row["squadra_modulo"] = lu_squadra.get("tactical_formation")
                out_row["squadra_allenatore"] = lu_squadra.get("coach_name")
                if lu_squadra.get("tactical_formation"):
                    n_match_modulo_squadra += 1
                if lu_squadra.get("coach_name"):
                    n_match_allenatore_squadra += 1
            else:
                out_row["squadra_modulo"] = None
                out_row["squadra_allenatore"] = None

            # --- modulo/allenatore avversario ---
            avv_lu = norm_team_lineups(avversario)
            lu_avversario = trova_lineup(lineups, stagione, data, avv_lu)
            if lu_avversario:
                out_row["avversario_modulo"] = lu_avversario.get("tactical_formation")
                out_row["avversario_allenatore"] = lu_avversario.get("coach_name")
                if lu_avversario.get("tactical_formation"):
                    n_match_modulo_avversario += 1
                if lu_avversario.get("coach_name"):
                    n_match_allenatore_avversario += 1
            else:
                out_row["avversario_modulo"] = None
                out_row["avversario_allenatore"] = None

            # --- classifica dinamica squadra ---
            cl_squadra = classifica.get((stagione, giornata, squadra))
            if cl_squadra:
                out_row["squadra_posizione_pre"] = cl_squadra.get("posizione_pre")
                out_row["squadra_punti_pre"] = cl_squadra.get("punti_pre")
                out_row["squadra_diff_reti_pre"] = cl_squadra.get("diff_reti_pre")
                n_match_classifica_squadra += 1
            else:
                out_row["squadra_posizione_pre"] = None
                out_row["squadra_punti_pre"] = None
                out_row["squadra_diff_reti_pre"] = None

            # --- classifica dinamica avversario ---
            cl_avversario = classifica.get((stagione, giornata, avversario))
            if cl_avversario:
                out_row["avversario_posizione_pre"] = cl_avversario.get("posizione_pre")
                out_row["avversario_punti_pre"] = cl_avversario.get("punti_pre")
                out_row["avversario_diff_reti_pre"] = cl_avversario.get("diff_reti_pre")
                n_match_classifica_avversario += 1
            else:
                out_row["avversario_posizione_pre"] = None
                out_row["avversario_punti_pre"] = None
                out_row["avversario_diff_reti_pre"] = None

            writer.writerow(out_row)

    log.info("Righe totali: %d", n_rows)
    log.info("Modulo squadra: %d (%.1f%%)", n_match_modulo_squadra, 100 * n_match_modulo_squadra / n_rows)
    log.info("Modulo avversario: %d (%.1f%%)", n_match_modulo_avversario, 100 * n_match_modulo_avversario / n_rows)
    log.info("Allenatore squadra: %d (%.1f%%)", n_match_allenatore_squadra, 100 * n_match_allenatore_squadra / n_rows)
    log.info("Allenatore avversario: %d (%.1f%%)", n_match_allenatore_avversario, 100 * n_match_allenatore_avversario / n_rows)
    log.info("Classifica squadra: %d (%.1f%%)", n_match_classifica_squadra, 100 * n_match_classifica_squadra / n_rows)
    log.info("Classifica avversario: %d (%.1f%%)", n_match_classifica_avversario, 100 * n_match_classifica_avversario / n_rows)
    log.info("Dataset feature v2 scritto in %s", OUT_PATH)


if __name__ == "__main__":
    main()
