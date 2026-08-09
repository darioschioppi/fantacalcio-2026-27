#!/usr/bin/env python3
"""
Scraper statistiche aggregate di stagione da Lega Serie A (provider Deltatre).
API pubblica scoperta analizzando i bundle JS del sito legaseriea.it:
  https://seriea-api.prd.sdp.deltatre.digital/v1/serie-a/football/seasons/{seasonId}/stats/teams

ATTENZIONE: l'endpoint gemello /seasons/{seasonId}/stats/players esiste ma è
INATTENDIBILE - ignora il seasonId/teamId per il filtro effettivo dei dati e
mescola giocatori/statistiche di stagioni e squadre diverse (verificato: es.
Zaccagni compare come "Lazio, 34 presenze" nella richiesta per la stagione
2019-20, quando in realtà era al Verona; l'endpoint restituisce sempre la
prima pagina del suo pool interno indipendentemente dai filtri passati).
Per questo la funzione scarica_players() esiste ma NON viene più chiamata da
main(): usare solo scarica_teams(), che invece è verificato corretto (20
squadre esatte per stagione, valori plausibili e coerenti tra stagioni).

Produce, per ogni stagione 2015-16..2025-26:
  work/data/lega_stats_teams_<stagione>.csv    (una riga per squadra)
più un aggregato finale lega_stats_teams_storico_2015_2026.csv.

Uso:
  python3 scrape_lega_stats.py
"""
import csv
import json
import time
import random
import logging
from pathlib import Path

import requests

API_BASE = "https://seriea-api.prd.sdp.deltatre.digital/v1/serie-a/football"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "lega_scrape_log.txt"

# stagione (label leggibile usato nei nostri file) -> seasonId interno Deltatre
SEASON_IDS = {
    "2015-16": "serie-a::Football_Season::f79cca154ea44ff88c461470e25dee9f",
    "2016-17": "serie-a::Football_Season::9a3082a64aae46a7829a9e3bbac8ec27",
    "2017-18": "serie-a::Football_Season::5e1b217da037428e865ab5b748ceed36",
    "2018-19": "serie-a::Football_Season::6e598253b5cc42ea9b79ab8e1ba6807a",
    "2019-20": "serie-a::Football_Season::1c0d03c3412e4e23a2ea056edc28ef16",
    "2020-21": "serie-a::Football_Season::860d1e2ac531405d93f70b16a19d140c",
    "2021-22": "serie-a::Football_Season::4c67f7c66d484e559a65857eb5a7cbeb",
    "2022-23": "serie-a::Football_Season::65f4d59dedbb43b68197b0ff0529fa21",
    "2023-24": "serie-a::Football_Season::104a84bc07f641e685f70a850c6399eb",
    "2024-25": "serie-a::Football_Season::1e32f55e98fc408a9d1fc27c0ba43243",
    "2025-26": "serie-a::Football_Season::5f0e080fc3a44073984b75b3a8e06a8a",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

PLAYER_META_FIELDS = [
    "playerId", "providerId", "bibNumber", "roleLabel", "role",
    "mediaFirstName", "mediaLastName", "shirtName", "shortName",
    "displayName", "nationality", "nationalityIsoCode",
]
TEAM_META_FIELDS = ["teamId", "providerId", "shortName", "officialName", "acronymName"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("lega_scraper")


def get_json(url, session, params=None, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, params=params, timeout=20)
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("Errore rete su %s (tentativo %d): %s (retry in %ds)", url, attempt, e, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = 5 * attempt
            log.warning("Rate limited su %s, attendo %ds", url, wait)
            time.sleep(wait)
            continue
        log.warning("HTTP %d su %s (tentativo %d)", resp.status_code, url, attempt)
        time.sleep(2 ** attempt)
    log.error("Fallito il download di %s dopo %d tentativi", url, max_retries)
    return None


def flatten_stats(item, meta_fields, stagione, entity_label):
    """Trasforma un elemento (player o team) con item['stats'] = [{statsId, statsValue}, ...]
    in un dict piatto: metadati + una colonna per ogni statsId incontrato."""
    row = {"stagione": stagione}
    if entity_label == "player":
        team = item.get("team") or {}
        row["team_id"] = team.get("teamId")
        row["team_name"] = team.get("shortName") or team.get("officialName")
    for f in meta_fields:
        row[f] = item.get(f)
    for s in item.get("stats", []):
        stats_id = s.get("statsId")
        if stats_id:
            row[f"stat__{stats_id}"] = s.get("statsValue")
    return row


def scarica_players(stagione, season_id, session):
    csv_path = DATA_DIR / f"lega_stats_players_{stagione}.csv"
    if csv_path.exists():
        log.info("Players %s: file già presente, salto (%s)", stagione, csv_path.name)
        return

    all_rows = []
    page = 1
    while True:
        url = f"{API_BASE}/seasons/{season_id}/stats/players"
        data = get_json(url, session, params={"page": page})
        if data is None:
            log.error("Players %s pagina %d: fallito, interrotto", stagione, page)
            break
        players = data.get("players", [])
        for p in players:
            all_rows.append(flatten_stats(p, PLAYER_META_FIELDS, stagione, "player"))
        pagination = data.get("pagination", {})
        log.info("Players %s pagina %d/%s: %d giocatori", stagione, page,
                  pagination.get("totalPages"), len(players))
        if pagination.get("isLastPage", True) or not players:
            break
        page += 1
        time.sleep(0.8 + random.random() * 0.4)

    if not all_rows:
        log.warning("Players %s: nessuna riga raccolta", stagione)
        return

    # Unione di tutte le colonne stat__* incontrate su tutte le righe (schema dinamico)
    fieldnames = ["stagione", "team_id", "team_name"] + PLAYER_META_FIELDS
    stat_cols = sorted({k for row in all_rows for k in row if k.startswith("stat__")})
    fieldnames += stat_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    log.info("Players %s: salvate %d righe in %s (%d colonne stat)",
              stagione, len(all_rows), csv_path.name, len(stat_cols))


def scarica_teams(stagione, season_id, session):
    csv_path = DATA_DIR / f"lega_stats_teams_{stagione}.csv"
    if csv_path.exists():
        log.info("Teams %s: file già presente, salto (%s)", stagione, csv_path.name)
        return

    url = f"{API_BASE}/seasons/{season_id}/stats/teams"
    data = get_json(url, session)
    if data is None:
        log.error("Teams %s: fallito il download", stagione)
        return

    teams = data.get("teams", [])
    rows = [flatten_stats(t, TEAM_META_FIELDS, stagione, "team") for t in teams]
    if not rows:
        log.warning("Teams %s: nessuna riga raccolta", stagione)
        return

    fieldnames = ["stagione"] + TEAM_META_FIELDS
    stat_cols = sorted({k for row in rows for k in row if k.startswith("stat__")})
    fieldnames += stat_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Teams %s: salvate %d righe in %s (%d colonne stat)",
              stagione, len(rows), csv_path.name, len(stat_cols))


def concatena_aggregato(prefix, stagioni):
    """Concatena i CSV per stagione in un unico file aggregato, gestendo
    colonne stat__* diverse tra stagioni (unione delle colonne, celle vuote se assenti)."""
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

    # NOTA: scarica_players() non viene chiamata - vedi docstring del modulo
    # per il motivo (endpoint stats/players inattendibile).
    for stagione, season_id in SEASON_IDS.items():
        scarica_teams(stagione, season_id, session)
        time.sleep(0.8 + random.random() * 0.4)

    n_teams = concatena_aggregato("lega_stats_teams", SEASON_IDS.keys())

    log.info("Completato. Righe teams aggregato: %d", n_teams)


if __name__ == "__main__":
    main()
