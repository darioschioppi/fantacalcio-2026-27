#!/usr/bin/env python3
"""
Costruisce il dataset per il modello PREVISIONALE (predice il voto PRIMA
che la partita si giochi), correggendo il problema di leakage individuato
analizzando il modello v1/v2: quelle versioni usavano come feature
statistiche della PARTITA STESSA da predire (understat_goals/xG/shots/...,
gol_fatti/subiti, assist, cartellini, ecc.) o totali di FINE STAGIONE
applicati a ogni giornata (squadra_total-points/goals/...).

Parte da work/data/voti_storici_2015_2026.csv (voto/target + colonne
identificative/contesto base: giornata, squadre, ruolo) e aggiunge SOLO
feature calcolate con informazione disponibile PRIMA del fischio d'inizio:

  - work/data/squadra_form_dinamica_storico_2015_2026.csv
    (forma rolling squadra/avversario, ultime 3-5 partite, no leakage -
    vedi build_squadra_form_dinamica.py)
  - work/data/player_lag_features_storico_2015_2026.csv
    (storia recente del giocatore: voto/xG/xA/shots/minuti lag, presenze
    cumulate - vedi build_player_lag_features.py)
  - work/data/classifica_dinamica_storico_2015_2026.csv
    (classifica pre-partita squadra/avversario - già esistente, nessuna
    modifica, join diretto su stagione+giornata+squadra)
  - work/data/lineups_storico_2015_2026.csv
    (modulo tattico e allenatore squadra/avversario - già esistente,
    stesso join ±1 giorno di build_feature_dataset_v2.py)

NON include (deliberatamente escluse - sono l'esito della partita che si
vuole predire o le sue conseguenze dirette): gol_fatti, gol_subiti,
autogol, rigori_segnati/sbagliati/parati, assist, ammonizione, espulsione,
mvp, gol_casa, gol_ospite, e nessuna colonna understat_* relativa alla
partita corrente (quelle sono confluite nei LAG, non nella riga stessa).

NOTA su squadra_modulo/avversario_modulo: la formazione ufficiale viene
comunicata di norma un'ora circa prima del calcio d'inizio - non è
derivata dal risultato della partita, ma nemmeno disponibile con largo
anticipo (non utilizzabile per un pronostico "a bocce ferme" con giorni di
anticipo). Si include comunque come feature legittima "pre-fischio
d'inizio", con questa avvertenza esplicita riportata nel report.

Join: tutti i join usano (stagione, giornata, squadra) per classifica/forma
(nessuna ambiguità, stessa fonte) e (stagione, data ±1 giorno, squadra
normalizzata) per lineups (stesso pattern già validato in
build_feature_dataset_v2.py).

Output: work/data/feature_dataset_v3.csv
"""
import csv
import logging
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
FORMA_PATH = DATA_DIR / "squadra_form_dinamica_storico_2015_2026.csv"
LAG_PATH = DATA_DIR / "player_lag_features_storico_2015_2026.csv"
CLASSIFICA_PATH = DATA_DIR / "classifica_dinamica_storico_2015_2026.csv"
LINEUPS_PATH = DATA_DIR / "lineups_storico_2015_2026.csv"
OUT_PATH = DATA_DIR / "feature_dataset_v3.csv"
LOG_PATH = DATA_DIR / "build_feature_dataset_v3_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("build_features_v3")

# stessa mappa di build_feature_dataset_v2.py
TEAM_NAME_MAP_LINEUPS = {
    "Chievo": "Chievoverona",
    "SPAL": "Spal",
}

# Colonne di voti_storici da mantenere come base (identificativi + contesto
# legittimo pre-partita: NON gol_casa/gol_ospite/gol_fatti/... - quelli sono
# esiti della partita)
BASE_COLS = [
    "stagione", "giornata", "match_id", "data",
    "squadra_casa", "squadra_ospite",
    "squadra_giocatore", "ruolo", "nome_giocatore", "player_id",
    "fonte_voto", "voto", "fantavoto", "senza_voto",
]

FORMA_COLS = [
    "forma3_punti_mean", "forma3_gf_mean", "forma3_gs_mean",
    "forma3_xg_for_mean", "forma3_xg_against_mean", "forma3_n_partite",
    "forma5_punti_mean", "forma5_gf_mean", "forma5_gs_mean",
    "forma5_xg_for_mean", "forma5_xg_against_mean", "forma5_n_partite",
]

LAG_COLS = [
    "voto_lag_mean_3", "voto_lag_mean_5",
    "understat_xG_lag_mean_5", "understat_xA_lag_mean_5",
    "understat_shots_lag_mean_5", "understat_time_lag_mean_5",
    "presenze_cumulate",
]

CLASSIFICA_COLS = ["posizione_pre", "punti_pre", "diff_reti_pre"]

# Colonne esplicitamente VIETATE (leakage) - controllo automatico a fine script
COLONNE_VIETATE = {
    "understat_goals", "understat_assists", "understat_shots", "understat_key_passes",
    "understat_xG", "understat_xA", "understat_xGChain", "understat_xGBuildup",
    "understat_time", "understat_position", "understat_own_goals",
    "understat_yellow_card", "understat_red_card",
    "gol_fatti", "gol_subiti", "autogol", "rigori_segnati", "rigori_sbagliati",
    "rigori_parati", "assist", "ammonizione", "espulsione", "mvp",
    "gol_casa", "gol_ospite",
}


def norm_team_lineups(name):
    return TEAM_NAME_MAP_LINEUPS.get(name, name)


def carica_forma():
    """dict {(stagione, giornata, squadra): row}"""
    forma = {}
    with open(FORMA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["stagione"], row["giornata"], row["squadra"])
            forma[key] = row
    log.info("Caricate %d righe forma dinamica squadra", len(forma))
    return forma


def carica_lag():
    """dict {(match_id, player_id): row}"""
    lag = {}
    with open(LAG_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["match_id"], row["player_id"])
            lag[key] = row
    log.info("Caricate %d righe lag giocatore", len(lag))
    return lag


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


def main():
    forma = carica_forma()
    lag = carica_lag()
    classifica = carica_classifica()
    lineups = carica_lineups()

    n_rows = 0
    n_match_forma_sq = 0
    n_match_forma_avv = 0
    n_match_lag = 0
    n_match_classifica_sq = 0
    n_match_classifica_avv = 0
    n_match_modulo_sq = 0
    n_match_modulo_avv = 0
    n_match_allenatore_sq = 0
    n_match_allenatore_avv = 0

    nuove_colonne = (
        [f"squadra_{c}" for c in FORMA_COLS] +
        [f"avversario_{c}" for c in FORMA_COLS] +
        LAG_COLS +
        [f"squadra_{c}" for c in CLASSIFICA_COLS] +
        [f"avversario_{c}" for c in CLASSIFICA_COLS] +
        ["squadra_modulo", "avversario_modulo", "squadra_allenatore", "avversario_allenatore"]
    )

    with open(VOTI_PATH, newline="", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = BASE_COLS + nuove_colonne
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
            match_id = row["match_id"]
            player_id = row["player_id"]

            out_row = {c: row.get(c) for c in BASE_COLS}

            # --- forma dinamica squadra ---
            f_sq = forma.get((stagione, giornata, squadra))
            if f_sq:
                n_match_forma_sq += 1
                for c in FORMA_COLS:
                    out_row[f"squadra_{c}"] = f_sq.get(c)
            else:
                for c in FORMA_COLS:
                    out_row[f"squadra_{c}"] = None

            # --- forma dinamica avversario ---
            f_avv = forma.get((stagione, giornata, avversario))
            if f_avv:
                n_match_forma_avv += 1
                for c in FORMA_COLS:
                    out_row[f"avversario_{c}"] = f_avv.get(c)
            else:
                for c in FORMA_COLS:
                    out_row[f"avversario_{c}"] = None

            # --- lag giocatore ---
            l = lag.get((match_id, player_id))
            if l:
                n_match_lag += 1
                for c in LAG_COLS:
                    out_row[c] = l.get(c)
            else:
                for c in LAG_COLS:
                    out_row[c] = None

            # --- classifica dinamica squadra/avversario ---
            cl_sq = classifica.get((stagione, giornata, squadra))
            if cl_sq:
                n_match_classifica_sq += 1
                for c in CLASSIFICA_COLS:
                    out_row[f"squadra_{c}"] = cl_sq.get(c)
            else:
                for c in CLASSIFICA_COLS:
                    out_row[f"squadra_{c}"] = None

            cl_avv = classifica.get((stagione, giornata, avversario))
            if cl_avv:
                n_match_classifica_avv += 1
                for c in CLASSIFICA_COLS:
                    out_row[f"avversario_{c}"] = cl_avv.get(c)
            else:
                for c in CLASSIFICA_COLS:
                    out_row[f"avversario_{c}"] = None

            # --- modulo/allenatore squadra/avversario ---
            lu_sq = trova_lineup(lineups, stagione, data, norm_team_lineups(squadra))
            if lu_sq:
                out_row["squadra_modulo"] = lu_sq.get("tactical_formation")
                out_row["squadra_allenatore"] = lu_sq.get("coach_name")
                if lu_sq.get("tactical_formation"):
                    n_match_modulo_sq += 1
                if lu_sq.get("coach_name"):
                    n_match_allenatore_sq += 1
            else:
                out_row["squadra_modulo"] = None
                out_row["squadra_allenatore"] = None

            lu_avv = trova_lineup(lineups, stagione, data, norm_team_lineups(avversario))
            if lu_avv:
                out_row["avversario_modulo"] = lu_avv.get("tactical_formation")
                out_row["avversario_allenatore"] = lu_avv.get("coach_name")
                if lu_avv.get("tactical_formation"):
                    n_match_modulo_avv += 1
                if lu_avv.get("coach_name"):
                    n_match_allenatore_avv += 1
            else:
                out_row["avversario_modulo"] = None
                out_row["avversario_allenatore"] = None

            writer.writerow(out_row)

    log.info("Righe totali: %d", n_rows)
    log.info("Forma squadra: %d (%.1f%%)", n_match_forma_sq, 100 * n_match_forma_sq / n_rows)
    log.info("Forma avversario: %d (%.1f%%)", n_match_forma_avv, 100 * n_match_forma_avv / n_rows)
    log.info("Lag giocatore: %d (%.1f%%)", n_match_lag, 100 * n_match_lag / n_rows)
    log.info("Classifica squadra: %d (%.1f%%)", n_match_classifica_sq, 100 * n_match_classifica_sq / n_rows)
    log.info("Classifica avversario: %d (%.1f%%)", n_match_classifica_avv, 100 * n_match_classifica_avv / n_rows)
    log.info("Modulo squadra: %d (%.1f%%)", n_match_modulo_sq, 100 * n_match_modulo_sq / n_rows)
    log.info("Modulo avversario: %d (%.1f%%)", n_match_modulo_avv, 100 * n_match_modulo_avv / n_rows)
    log.info("Allenatore squadra: %d (%.1f%%)", n_match_allenatore_sq, 100 * n_match_allenatore_sq / n_rows)
    log.info("Allenatore avversario: %d (%.1f%%)", n_match_allenatore_avv, 100 * n_match_allenatore_avv / n_rows)

    # --- Verifica automatica anti-leakage ---
    colonne_finali = set(fieldnames)
    intersezione = colonne_finali & COLONNE_VIETATE
    if intersezione:
        log.error("LEAKAGE RILEVATO! Colonne vietate presenti nell'output: %s", intersezione)
        raise SystemExit(1)
    log.info("Verifica anti-leakage OK: nessuna colonna vietata presente tra le %d colonne finali", len(colonne_finali))

    log.info("Dataset feature v3 scritto in %s", OUT_PATH)


if __name__ == "__main__":
    main()
