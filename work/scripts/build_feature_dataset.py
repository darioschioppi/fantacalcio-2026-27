#!/usr/bin/env python3
"""
Costruisce il dataset di feature per il modello ML che predice la
valutazione fantacalcio, unendo:
  - work/data/voti_storici_2015_2026.csv                    (target: voto/fantavoto per giocatore/partita/fonte)
  - work/data/lega_stats_teams_storico_2015_2026.csv         (feature di contesto squadra per stagione, Lega Serie A)
  - work/data/understat_player_match_stats_storico_2015_2026.csv (feature INDIVIDUALI giocatore per singola partita: xG, xA, tiri, ecc.)
  - work/data/player_name_mapping.csv                        (mappa nome_fantacalcio -> player_id Understat)

NOTA STORICA: le statistiche individuali di Lega Serie A (endpoint
stats/players) sono state scartate perché inattendibili - vedi
work/scripts/scrape_lega_stats.py. Le statistiche individuali vengono
quindi da Understat (vedi work/scripts/scrape_understat.py e
work/scripts/build_player_name_mapping.py per la strategia di matching
nomi, che ha una copertura del 99.5% delle combinazioni partita/squadra;
il residuo ~0.5% mancante sono principalmente partite rinviate/recuperate
con date discordanti tra le due fonti - non rincorse oltre, impatto
trascurabile).

Il join usa:
  - (stagione, nome_squadra_normalizzato) per il contesto SQUADRA (Lega Serie A)
  - (stagione, squadra, nome_giocatore) -> player_id_understat via la mappa,
    poi (stagione, team_title, player_id, data ± 1 giorno) per le stats
    INDIVIDUALI Understat della singola partita.

Output: work/data/feature_dataset_v1.csv
(v0 = solo contesto squadra; v1 = squadra + individuali Understat)
"""
import csv
import re
import logging
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
LEGA_TEAMS_PATH = DATA_DIR / "lega_stats_teams_storico_2015_2026.csv"
UNDERSTAT_PATH = DATA_DIR / "understat_player_match_stats_storico_2015_2026.csv"
MAPPING_PATH = DATA_DIR / "player_name_mapping.csv"
OUT_PATH = DATA_DIR / "feature_dataset_v1.csv"
LOG_PATH = DATA_DIR / "build_feature_dataset_v1_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("build_features_v1")

# Mappa nomi squadra fantacalcio.it -> nome squadra Lega Serie A (shortName)
TEAM_NAME_MAP_LEGA = {
    "Chievo": "Chievoverona",
    "SPAL": "Spal",
}

# Mappa nomi squadra fantacalcio.it -> nome squadra Understat (team_title)
TEAM_NAME_MAP_UNDERSTAT = {
    "SPAL": "SPAL 2013",
    "Milan": "AC Milan",
    "Parma": "Parma Calcio 1913",
}

TEAM_STAT_COLS = [
    "stat__games-played",
    "stat__total-points",
    "stat__total-wins",
    "stat__total-draws",
    "stat__total-losses",
    "stat__goals",
    "stat__goals-against",
    "stat__expectedGoals",
    "stat__expectedGoalAgainst",
    "stat__home-wins",
    "stat__home-draws",
    "stat__home-losses",
    "stat__away-wins",
    "stat__away-draws",
    "stat__away-losses",
]

UNDERSTAT_STAT_COLS = [
    "position", "time", "goals", "own_goals", "assists", "shots",
    "key_passes", "xG", "xA", "xGChain", "xGBuildup", "yellow_card", "red_card",
]


def norm_team_lega(name):
    return TEAM_NAME_MAP_LEGA.get(name, name)


def norm_team_understat(name):
    return TEAM_NAME_MAP_UNDERSTAT.get(name, name)


def carica_team_stats():
    team_stats = {}
    with open(LEGA_TEAMS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        available_cols = [c for c in TEAM_STAT_COLS if c in reader.fieldnames]
        missing = set(TEAM_STAT_COLS) - set(available_cols)
        if missing:
            log.warning("Colonne stat squadra (Lega) non trovate nel CSV: %s", missing)
        for row in reader:
            key = (row["stagione"], row["shortName"])
            team_stats[key] = {c: row.get(c) for c in available_cols}
    log.info("Caricate statistiche squadra (Lega) per %d combinazioni (stagione, squadra)", len(team_stats))
    return team_stats, available_cols


def carica_name_mapping():
    """dict {(stagione, squadra, nome_fantacalcio): player_id_understat}"""
    mapping = {}
    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["stagione"], row["squadra"], row["nome_fantacalcio"])
            mapping[key] = row["player_id_understat"]
    log.info("Caricata mappa nomi giocatore: %d coppie (stagione, squadra, nome)", len(mapping))
    return mapping


def carica_understat_stats():
    """dict {(stagione, team_title, player_id, data): stats_row}
    data = solo YYYY-MM-DD (senza ora)."""
    stats = {}
    with open(UNDERSTAT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = (row["match_date"] or "")[:10]
            key = (row["stagione"], row["team_title"], row["player_id"], data)
            stats[key] = row
    log.info("Caricate statistiche partita Understat: %d righe", len(stats))
    return stats


def trova_stats_understat(understat_stats, stagione, team_title, player_id, data_str):
    """Cerca la riga Understat per (stagione, team_title, player_id, data);
    se non trovata esattamente, prova ±1 giorno (scarto di fuso orario UTC
    per partite serali, stesso problema riscontrato nel matching nomi)."""
    if not player_id or not data_str:
        return None
    key = (stagione, team_title, player_id, data_str)
    if key in understat_stats:
        return understat_stats[key]
    try:
        y, m, d = (int(x) for x in data_str.split("-"))
        base = date(y, m, d)
    except ValueError:
        return None
    for delta in (1, -1):
        alt_data = (base + timedelta(days=delta)).isoformat()
        alt_key = (stagione, team_title, player_id, alt_data)
        if alt_key in understat_stats:
            return understat_stats[alt_key]
    return None


def main():
    team_stats, team_stat_cols = carica_team_stats()
    name_mapping = carica_name_mapping()
    understat_stats = carica_understat_stats()

    n_rows = 0
    n_matched_team = 0
    n_matched_player = 0
    unmatched_teams = set()

    squadra_cols = [f"squadra_{c.replace('stat__', '')}" for c in team_stat_cols]
    player_cols = [f"understat_{c}" for c in UNDERSTAT_STAT_COLS]

    with open(VOTI_PATH, newline="", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + squadra_cols + player_cols
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            n_rows += 1
            stagione = row["stagione"]
            squadra_giocatore = row["squadra_giocatore"]
            out_row = dict(row)

            # --- contesto squadra (Lega Serie A) ---
            team_lega = norm_team_lega(squadra_giocatore)
            key_lega = (stagione, team_lega)
            stats_squadra = team_stats.get(key_lega)
            if stats_squadra:
                n_matched_team += 1
                for c in team_stat_cols:
                    out_row[f"squadra_{c.replace('stat__', '')}"] = stats_squadra.get(c)
            else:
                unmatched_teams.add(key_lega)
                for c in team_stat_cols:
                    out_row[f"squadra_{c.replace('stat__', '')}"] = None

            # --- statistiche individuali giocatore/partita (Understat) ---
            team_understat = norm_team_understat(squadra_giocatore)
            map_key = (stagione, team_understat, row["nome_giocatore"])
            player_id = name_mapping.get(map_key)
            stats_giocatore = trova_stats_understat(
                understat_stats, stagione, team_understat, player_id, row["data"]
            )
            if stats_giocatore:
                n_matched_player += 1
                for c in UNDERSTAT_STAT_COLS:
                    out_row[f"understat_{c}"] = stats_giocatore.get(c)
            else:
                for c in UNDERSTAT_STAT_COLS:
                    out_row[f"understat_{c}"] = None

            writer.writerow(out_row)

    log.info("Righe totali: %d", n_rows)
    log.info("Con contesto squadra agganciato: %d (%.1f%%)", n_matched_team, 100 * n_matched_team / n_rows if n_rows else 0)
    log.info("Con statistiche individuali Understat agganciate: %d (%.1f%%)", n_matched_player, 100 * n_matched_player / n_rows if n_rows else 0)
    if unmatched_teams:
        log.warning("Combinazioni (stagione, squadra) senza match nelle stats Lega (%d): %s",
                    len(unmatched_teams), sorted(unmatched_teams))
    log.info("Dataset feature scritto in %s", OUT_PATH)


if __name__ == "__main__":
    main()
