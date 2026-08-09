#!/usr/bin/env python3
"""
Scraper formazioni tattiche e allenatori da Lega Serie A (provider Deltatre),
stessa API già usata in scrape_lega_stats.py e verificata affidabile:
  https://seriea-api.prd.sdp.deltatre.digital/v1/serie-a/football/seasons/{seasonId}/matches
  https://seriea-api.prd.sdp.deltatre.digital/v1/serie-a/football/seasons/{seasonId}/matches/{matchId}/lineups

/matches (senza paginazione, un'unica chiamata restituisce tutte le ~380
partite della stagione con il loro matchId Deltatre) fornisce l'elenco
partite; /lineups per singolo matchId fornisce, per casa e ospite:
  - tacticalFormation (es. "4-3-3") - verificato presente per TUTTE le 11
    stagioni 2015-16..2025-26.
  - staff (lista con roleLabel, tra cui "Head Coach") - verificato PRESENTE
    SOLO dalla stagione 2023-24 in avanti; nelle stagioni precedenti il
    campo è sempre una lista vuota. Questo è un limite noto della fonte,
    non un bug: si salva coach_name=None dove non disponibile.

Produce, per ogni stagione 2015-16..2025-26:
  work/data/lineups_<stagione>.csv   (una riga per squadra per partita, quindi
                                       2 righe per partita)
più un aggregato finale lineups_storico_2015_2026.csv.

Uso:
  python3 scrape_lineups.py
"""
import csv
import time
import random
import logging
from pathlib import Path

import requests

API_BASE = "https://seriea-api.prd.sdp.deltatre.digital/v1/serie-a/football"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "lineups_scrape_log.txt"

# stesso mapping stagione -> seasonId di scrape_lega_stats.py / scrape_understat.py
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

FIELDNAMES = [
    "stagione", "match_id", "match_date", "team_id", "team_title",
    "h_a", "opponent_title", "tactical_formation", "coach_name",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("lineups_scraper")


def get_json(url, session, max_retries=4):
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("Errore rete su %s (tentativo %d): %s (retry in %ds)", url, attempt, e, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = 8 * attempt
            log.warning("Rate limited su %s, attendo %ds", url, wait)
            time.sleep(wait)
            continue
        log.warning("HTTP %d su %s (tentativo %d)", resp.status_code, url, attempt)
        time.sleep(2 ** attempt)
    log.error("Fallito il download di %s dopo %d tentativi", url, max_retries)
    return None


def scarica_match_ids(stagione, season_id, session):
    url = f"{API_BASE}/seasons/{season_id}/matches"
    data = get_json(url, session)
    if data is None:
        return []
    matches = data.get("matches", [])
    log.info("%s: %d partite trovate via /matches", stagione, len(matches))
    return matches


def normalizza_nome_persona(nome):
    """Normalizza un nome/cognome in Title Case gestendo correttamente
    apostrofi (es. "d'aversa" -> "D'Aversa") e accenti (preservati)."""
    if not nome:
        return nome
    return nome.strip().title()


def estrai_coach(staff_list):
    """Estrae il nome dell'allenatore in forma CANONICA e stabile.

    NOTA (bug scoperto in validazione a valle, corretto qui): il campo
    shortName/displayName della staff list è INCOERENTE tra i provider di
    dati sottostanti (kama vs opta): a volte "Nome Cognome" completo, a
    volte "N. Cognome" abbreviato, con/senza accenti (es. "Ivan Jurić" vs
    "I. Juric", "Daniele De Rossi" vs "D. De Rossi"). Usare quel campo
    grezzo produce lo stesso allenatore sotto più etichette diverse nel
    dataset, diluendo qualunque analisi raggruppata per allenatore.

    mediaFirstName/mediaLastName sono invece SEMPRE il nome completo,
    consistenti su tutte le stagioni verificate: si usano questi,
    normalizzati in Title Case, per costruire un nome canonico univoco.
    """
    for s in staff_list or []:
        if (s.get("roleLabel") or "").strip().lower() == "head coach":
            fn = normalizza_nome_persona(s.get("mediaFirstName"))
            ln = normalizza_nome_persona(s.get("mediaLastName"))
            nome_canonico = f"{fn or ''} {ln or ''}".strip()
            if nome_canonico:
                return nome_canonico
            # fallback se mediaFirstName/mediaLastName sono entrambi vuoti
            return s.get("shortName") or s.get("displayName")
    return None


def righe_da_lineup(stagione, match_id, match_date, lineup_data):
    righe = []
    home = lineup_data.get("home") or {}
    away = lineup_data.get("away") or {}
    if home:
        righe.append({
            "stagione": stagione,
            "match_id": match_id,
            "match_date": match_date,
            "team_id": home.get("teamId"),
            "team_title": home.get("shortName"),
            "h_a": "h",
            "opponent_title": away.get("shortName"),
            "tactical_formation": home.get("tacticalFormation"),
            "coach_name": estrai_coach(home.get("staff")),
        })
    if away:
        righe.append({
            "stagione": stagione,
            "match_id": match_id,
            "match_date": match_date,
            "team_id": away.get("teamId"),
            "team_title": away.get("shortName"),
            "h_a": "a",
            "opponent_title": home.get("shortName"),
            "tactical_formation": away.get("tacticalFormation"),
            "coach_name": estrai_coach(away.get("staff")),
        })
    return righe


def match_gia_salvati(csv_path):
    salvati = set()
    if not csv_path.exists():
        return salvati
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            salvati.add(row.get("match_id"))
    return salvati


def scarica_stagione(stagione, season_id, session):
    csv_path = DATA_DIR / f"lineups_{stagione}.csv"
    match_ids_salvati = match_gia_salvati(csv_path)

    matches = scarica_match_ids(stagione, season_id, session)
    if not matches:
        log.warning("%s: nessuna partita trovata, salto stagione", stagione)
        return 0, 0

    righe_da_scrivere = []
    n_ok = 0
    n_falliti = 0

    for m in matches:
        match_id = m.get("matchId")
        if not match_id or match_id in match_ids_salvati:
            continue

        url = f"{API_BASE}/seasons/{season_id}/matches/{match_id}/lineups"
        data = get_json(url, session)
        if data is None:
            n_falliti += 1
            continue

        righe = righe_da_lineup(stagione, match_id, m.get("matchDateUtc"), data)
        if righe:
            righe_da_scrivere.extend(righe)
            n_ok += 1
        else:
            n_falliti += 1

        time.sleep(1.0 + random.random() * 0.6)

    if righe_da_scrivere:
        file_esiste = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_esiste:
                writer.writeheader()
            for r in righe_da_scrivere:
                writer.writerow(r)

    log.info("%s: %d partite scaricate ok, %d fallite, %d righe scritte",
              stagione, n_ok, n_falliti, len(righe_da_scrivere))
    return n_ok, n_falliti


def concatena_aggregato(stagioni):
    files = [DATA_DIR / f"lineups_{s}.csv" for s in stagioni if (DATA_DIR / f"lineups_{s}.csv").exists()]
    if not files:
        return 0
    out_path = DATA_DIR / "lineups_storico_2015_2026.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        writer.writeheader()
        totale = 0
        for fp in files:
            with open(fp, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    writer.writerow(row)
                    totale += 1
    log.info("Aggregato scritto in %s (%d righe)", out_path.name, totale)
    return totale


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    riepilogo = {}
    for stagione, season_id in SEASON_IDS.items():
        ok, falliti = scarica_stagione(stagione, season_id, session)
        riepilogo[stagione] = {"partite_ok": ok, "partite_fallite": falliti}
        time.sleep(1.0 + random.random() * 0.5)

    totale = concatena_aggregato(SEASON_IDS.keys())

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n=== RIEPILOGO FINALE ===\n")
        for stagione, info in riepilogo.items():
            f.write(f"{stagione}: {info}\n")
        f.write(f"TOTALE RIGHE AGGREGATO: {totale}\n")

    log.info("Scraping lineups completato. Righe totali: %d", totale)


if __name__ == "__main__":
    main()
