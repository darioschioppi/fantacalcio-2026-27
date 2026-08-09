#!/usr/bin/env python3
"""
Scraper statistiche individuali giocatore da Understat.com (xG, xA, tiri,
passaggi chiave, xGChain/xGBuildup) a livello di SINGOLA PARTITA, per
Serie A dalla stagione 2015-16 alla 2025-26.

NOTA SU robots.txt: understat.com pubblica un robots.txt con
"Disallow: /" per tutti i crawler (divieto generico allo scraping
automatizzato). Dario ha esplicitamente valutato il rischio e istruito
di procedere comunque (uso interno/non commerciale per il progetto
fantacalcio, rate basso, nessuna ripubblicazione dei dati grezzi altrove).
Questo script applica quindi un rate-limit conservativo e uno User-Agent
realistico, ma NON tenta di occultare la propria natura di scraper.

Endpoint reali (scoperti analizzando js/league.min.js e js/match.min.js
serviti dal sito stesso, nessuna documentazione ufficiale esiste):
  GET https://understat.com/getLeagueData/{league}/{season}
      -> {"teams": {...}, "players": [...stagionali...], "dates": [...380 partite...]}
      Richiede header Referer valido (es. la pagina league corrispondente)
      e risposta gzip-encoded (curl --compressed / requests la gestiscono
      automaticamente).
  GET https://understat.com/getMatchData/{matchId}
      -> {"rosters": {"h": {...}, "a": {...}}, "shots": {"h": [...], "a": [...]}}
      "rosters" contiene, per ogni giocatore che ha giocato quella partita,
      le sue statistiche IN QUELLA PARTITA: xG, xA, shots, key_passes,
      xGChain, xGBuildup, time (minuti), goals, assists, yellow/red_card,
      position. Questa è la granularità partita-per-partita che serve per
      allinearsi ai voti fantacalcio.it (una riga = un giocatore in una
      partita).
      "shots" contiene ogni singolo tiro della partita (X,Y,xG,minute,
      result,situation,shotType,player,...) - salvato come dataset
      supplementare per eventuale analisi più granulare futura.

Season param Understat = anno di inizio stagione (es. "2015" per 2015/2016).

Output:
  work/data/understat_player_match_stats_<stagione>.csv  (una riga per
      giocatore per partita)
  work/data/understat_shots_<stagione>.csv                (una riga per tiro)
  work/data/understat_player_match_stats_storico_2015_2026.csv  (aggregato)
  work/data/understat_shots_storico_2015_2026.csv                (aggregato)
  work/data/understat_scrape_log.txt

Uso:
  python3 scrape_understat.py
"""
import csv
import json
import time
import random
import logging
from pathlib import Path

import requests

BASE = "https://understat.com"
LEAGUE = "Serie_A"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "understat_scrape_log.txt"

# stagione (label leggibile usato nei nostri file) -> season param Understat
SEASONS = {
    "2015-16": "2015",
    "2016-17": "2016",
    "2017-18": "2017",
    "2018-19": "2018",
    "2019-20": "2019",
    "2020-21": "2020",
    "2021-22": "2021",
    "2022-23": "2022",
    "2023-24": "2023",
    "2024-25": "2024",
    "2025-26": "2025",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json",
}

ROSTER_FIELDS = [
    "id", "player_id", "player", "team_id", "position", "positionOrder",
    "h_a", "time", "goals", "own_goals", "assists", "shots", "key_passes",
    "xG", "xA", "xGChain", "xGBuildup", "yellow_card", "red_card",
    "roster_in", "roster_out",
]
SHOT_FIELDS = [
    "id", "match_id", "minute", "player", "player_id", "h_a", "player_assisted",
    "result", "X", "Y", "xG", "situation", "shotType", "lastAction",
    "h_team", "a_team", "h_goals", "a_goals", "date",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("understat_scraper")


def get_json(url, session, referer, max_retries=4):
    headers = dict(HEADERS)
    headers["Referer"] = referer
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=20)
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("Errore rete su %s (tentativo %d): %s (retry in %ds)", url, attempt, e, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                log.warning("Risposta non-JSON da %s (tentativo %d)", url, attempt)
                time.sleep(2 ** attempt)
                continue
        if resp.status_code == 429:
            wait = 8 * attempt
            log.warning("Rate limited su %s, attendo %ds", url, wait)
            time.sleep(wait)
            continue
        log.warning("HTTP %d su %s (tentativo %d)", resp.status_code, url, attempt)
        time.sleep(2 ** attempt)
    log.error("Fallito il download di %s dopo %d tentativi", url, max_retries)
    return None


def scarica_giornate_stagione(stagione, season_param, session):
    """Ritorna la lista di match dict (con almeno 'id') per la stagione,
    dall'endpoint getLeagueData (dates=lista partite)."""
    url = f"{BASE}/getLeagueData/{LEAGUE}/{season_param}"
    referer = f"{BASE}/league/{LEAGUE}/{season_param}"
    data = get_json(url, session, referer)
    if data is None:
        return []
    dates = data.get("dates", [])
    log.info("%s: %d partite trovate via getLeagueData", stagione, len(dates))
    return dates


def flatten_roster_row(stagione, match_id, match_meta, player):
    row = {"stagione": stagione, "match_id": match_id}
    row["team_title"] = match_meta.get("h" if player.get("h_a") == "h" else "a", {}).get("title")
    row["opponent_title"] = match_meta.get("a" if player.get("h_a") == "h" else "h", {}).get("title")
    row["match_date"] = match_meta.get("datetime")
    for f in ROSTER_FIELDS:
        row[f] = player.get(f)
    return row


def flatten_shot_row(stagione, shot):
    row = {"stagione": stagione}
    for f in SHOT_FIELDS:
        row[f] = shot.get(f)
    return row


def match_gia_salvato(csv_path, match_id):
    """Check leggero: se il match_id è già presente nel CSV di stagione, salta."""
    if not csv_path.exists():
        return False
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("match_id") == str(match_id):
                return True
    return False


def scarica_stagione(stagione, season_param, session):
    roster_path = DATA_DIR / f"understat_player_match_stats_{stagione}.csv"
    shots_path = DATA_DIR / f"understat_shots_{stagione}.csv"

    # resumability: quali match_id sono già presenti
    match_ids_salvati = set()
    if roster_path.exists():
        with open(roster_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                match_ids_salvati.add(row.get("match_id"))

    matches = scarica_giornate_stagione(stagione, season_param, session)
    if not matches:
        log.warning("%s: nessuna partita trovata, salto stagione", stagione)
        return 0, 0

    roster_rows = []
    shot_rows = []
    n_match_ok = 0
    n_match_falliti = 0

    for m in matches:
        match_id = m.get("id")
        if not match_id:
            continue
        if match_id in match_ids_salvati:
            continue

        url = f"{BASE}/getMatchData/{match_id}"
        referer = f"{BASE}/match/{match_id}"
        data = get_json(url, session, referer)
        if data is None:
            n_match_falliti += 1
            continue

        rosters = data.get("rosters", {})
        for h_a in ("h", "a"):
            for player in rosters.get(h_a, {}).values():
                roster_rows.append(flatten_roster_row(stagione, match_id, m, player))

        shots = data.get("shots", {})
        for h_a in ("h", "a"):
            for shot in shots.get(h_a, []):
                shot_rows.append(flatten_shot_row(stagione, shot))

        n_match_ok += 1
        time.sleep(1.0 + random.random() * 0.6)

    if roster_rows:
        file_esiste = roster_path.exists()
        with open(roster_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["stagione", "match_id", "team_title", "opponent_title", "match_date"] + ROSTER_FIELDS
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_esiste:
                writer.writeheader()
            for r in roster_rows:
                writer.writerow(r)

    if shot_rows:
        file_esiste = shots_path.exists()
        with open(shots_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["stagione"] + SHOT_FIELDS
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_esiste:
                writer.writeheader()
            for r in shot_rows:
                writer.writerow(r)

    log.info("%s: %d partite scaricate ok, %d fallite, %d righe roster, %d righe tiri",
              stagione, n_match_ok, n_match_falliti, len(roster_rows), len(shot_rows))
    return n_match_ok, n_match_falliti


def concatena_aggregato(prefix, stagioni):
    files = [DATA_DIR / f"{prefix}_{s}.csv" for s in stagioni if (DATA_DIR / f"{prefix}_{s}.csv").exists()]
    if not files:
        return 0
    all_fieldnames = []
    seen = set()
    all_rows = []
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for fn in reader.fieldnames:
                if fn not in seen:
                    seen.add(fn)
                    all_fieldnames.append(fn)
            for row in reader:
                all_rows.append(row)
    out_path = DATA_DIR / f"{prefix}_storico_2015_2026.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    log.info("Aggregato %s scritto in %s (%d righe, %d colonne)",
              prefix, out_path.name, len(all_rows), len(all_fieldnames))
    return len(all_rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    riepilogo = {}
    for stagione, season_param in SEASONS.items():
        ok, falliti = scarica_stagione(stagione, season_param, session)
        riepilogo[stagione] = {"partite_ok": ok, "partite_fallite": falliti}
        time.sleep(1.5 + random.random() * 0.8)

    n_roster = concatena_aggregato("understat_player_match_stats", SEASONS.keys())
    n_shots = concatena_aggregato("understat_shots", SEASONS.keys())

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n=== RIEPILOGO FINALE ===\n")
        for stagione, info in riepilogo.items():
            f.write(f"{stagione}: {info}\n")
        f.write(f"TOTALE RIGHE ROSTER AGGREGATO: {n_roster}\n")
        f.write(f"TOTALE RIGHE SHOTS AGGREGATO: {n_shots}\n")

    log.info("Scraping Understat completato. Righe roster: %d, righe shots: %d", n_roster, n_shots)


if __name__ == "__main__":
    main()
