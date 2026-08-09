#!/usr/bin/env python3
"""
Scraper voti storici fantacalcio.it
Scarica i voti/pagelle di ogni giornata di Serie A dalla pagina pubblica
https://www.fantacalcio.it/voti-fantacalcio-serie-a/{stagione}/{giornata}
(nessuna autenticazione richiesta) e produce:
  - work/data/voti_<stagione>.csv   (uno per stagione, resumable)
  - work/data/voti_storici_2015_2026.csv (aggregato finale)
  - work/data/scrape_log.txt (log/riepilogo)

Uso:
  python3 scrape_voti_fantacalcio.py
"""
import csv
import os
import re
import sys
import time
import random
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.fantacalcio.it/voti-fantacalcio-serie-a/{stagione}/{giornata}"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "scrape_log.txt"
AGGREGATE_PATH = DATA_DIR / "voti_storici_2015_2026.csv"

SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
GIORNATE = range(1, 39)  # 1..38

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}

FIELDNAMES = [
    "stagione", "giornata", "match_id", "data",
    "squadra_casa", "squadra_ospite", "gol_casa", "gol_ospite",
    "squadra_giocatore", "ruolo", "nome_giocatore", "player_id",
    "fonte_voto", "voto", "fantavoto", "senza_voto",
    "gol_fatti", "gol_subiti", "autogol",
    "rigori_segnati", "rigori_sbagliati", "rigori_parati",
    "assist", "ammonizione", "espulsione", "mvp",
]

RUOLO_MAP = {"p": "portiere", "d": "difensore", "c": "centrocampista", "a": "attaccante"}

SOURCE_ICON_MAP = {
    "ico-fc": "redazione",
    "ico-stats": "statistico",
    "ico-italy": "italia",
}

BONUS_TITLE_MAP = {
    "Gol segnati": "gol_fatti",
    "Gol subiti": "gol_subiti",
    "Autoreti": "autogol",
    "Rigori segnati": "rigori_segnati",
    "Rigori sbagliati": "rigori_sbagliati",
    "Rigori parati": "rigori_parati",
    "Assist": "assist",
    "Player of the match": "mvp",
}

NUM_RE = re.compile(r"-?\d+(?:[,.]\d+)?")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("scraper")


def parse_number(raw):
    """Convert an Italian comma-decimal string (or already-dotted) to float.
    Returns None if no usable number is found. Tolerant of anomalies."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    m = NUM_RE.search(raw)
    if not m:
        return None
    val = m.group(0).replace(",", ".")
    try:
        num = float(val)
    except ValueError:
        log.warning("Valore numerico non parsabile: %r", raw)
        return None
    # Sanity check: i voti fantacalcio sono tipicamente 0-15; fuori range logga ma non scarta
    if not (-1 <= num <= 200):
        log.warning("Valore numerico fuori range plausibile: %r -> %s", raw, num)
    return num


SV_SENTINEL = "55"  # valore sentinella usato dal sito per "Senza Voto" (non giocato/valutato)


def parse_voto(raw):
    """Come parse_number, ma tratta il valore sentinella '55' (usato da
    fantacalcio.it per indicare S.V. - Senza Voto, tipico di subentrati con
    minuti insufficienti) come assenza di voto (None) invece che come 55.0."""
    if raw is None:
        return None
    if raw.strip() == SV_SENTINEL:
        return None
    return parse_number(raw)


def scarica_giornata(stagione, giornata, session, max_retries=3):
    """GET della pagina voti. Ritorna l'HTML oppure None se non disponibile
    (redirect 301 a stagione corrente = stagione/giornata non esistente, o 404)."""
    url = BASE_URL.format(stagione=stagione, giornata=giornata)
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20, allow_redirects=False)
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("Errore rete %s/%s tentativo %d: %s (retry in %ds)",
                        stagione, giornata, attempt, e, wait)
            time.sleep(wait)
            continue

        if resp.status_code in (301, 302, 404):
            log.info("Stagione/giornata non disponibile: %s giornata %s (HTTP %d)",
                      stagione, giornata, resp.status_code)
            return None
        if resp.status_code == 200:
            return resp.text
        if resp.status_code == 429:
            wait = 5 * attempt
            log.warning("Rate limited su %s giornata %s, attendo %ds", stagione, giornata, wait)
            time.sleep(wait)
            continue
        # Altri status: retry con backoff
        wait = 2 ** attempt
        log.warning("HTTP %d su %s giornata %s tentativo %d (retry in %ds)",
                    resp.status_code, stagione, giornata, attempt, wait)
        time.sleep(wait)

    log.error("Fallito il download di %s giornata %s dopo %d tentativi", stagione, giornata, max_retries)
    return None


def estrai_match_id(href):
    """Estrae l'ID partita dall'URL di dettaglio, es.
    .../serie-a/calendario/38/2024-25/bologna-genoa/15694 -> 15694
    (ultimo segmento numerico dell'URL)."""
    if not href:
        return None
    m = re.search(r"/(\d+)/?$", href.rstrip("/"))
    return m.group(1) if m else None


def estrai_player_id(href):
    if not href:
        return None
    # .../team-slug/player-slug/PLAYER_ID/season
    m = re.search(r"/(\d+)/[^/]*$", href)
    if m:
        return m.group(1)
    m = re.search(r"/(\d+)/?$", href)
    return m.group(1) if m else None


def estrai_fonti_voto(grade_table):
    """Determina l'ordine delle fonti voto (redazione/statistico/italia) dalle
    icone presenti nella seconda riga del thead della tabella voti."""
    fonti = []
    thead_rows = grade_table.select("thead > tr")
    if len(thead_rows) < 2:
        return fonti
    icon_row = thead_rows[1]
    voto_th = icon_row.select_one("th")  # prima cella = colonna Voto/Fantavoto
    if not voto_th:
        return fonti
    for img in voto_th.select("img"):
        title = (img.get("title") or "").strip().lower()
        if "redazione" in title:
            fonti.append("redazione")
        elif "statistico" in title:
            fonti.append("statistico")
        elif "italia" in title:
            fonti.append("italia")
        else:
            fonti.append(title or "sconosciuta")
    return fonti


def parsa_giornata(html, stagione, giornata):
    """Ritorna una lista di dict (righe) estratte dalla pagina."""
    soup = BeautifulSoup(html, "lxml")
    righe = []

    li_matches = soup.select("li.match")

    if not li_matches:
        log.warning("Nessun match trovato per %s giornata %s (pagina vuota o struttura cambiata)",
                    stagione, giornata)
        return righe

    for li in li_matches:
        try:
            teams_id = li.get("data-teams-id", "")
            home_id, _, away_id = teams_id.partition("|")

            match_link = li.select_one("a.match-score")
            match_id = estrai_match_id(match_link["href"]) if match_link and match_link.has_attr("href") else None

            score_text = match_link.get_text(" ", strip=True) if match_link else ""
            score_parts = score_text.split("-")
            gol_casa = parse_number(score_parts[0]) if len(score_parts) == 2 else None
            gol_ospite = parse_number(score_parts[1]) if len(score_parts) == 2 else None

            date_meta = li.select_one("meta[itemprop=startDate]")
            data_match = date_meta["content"].strip() if date_meta and date_meta.has_attr("content") else None
        except Exception as e:
            log.warning("Errore parsing header match in %s giornata %s: %s", stagione, giornata, e)
            home_id = away_id = match_id = gol_casa = gol_ospite = data_match = None

        squadre_ids = [(home_id, "home"), (away_id, "away")]
        nomi_squadra = {}

        for team_id, ruolo_casa_ospite in squadre_ids:
            if not team_id:
                continue
            team_table = soup.select_one(f"#team-{team_id}")
            if not team_table:
                log.warning("team-table #team-%s non trovata per match %s (%s giornata %s)",
                            team_id, match_id, stagione, giornata)
                continue

            team_name_meta = team_table.select_one(
                "table.grades-table thead .team-info a meta[itemprop=name]"
            )
            squadra_giocatore = team_name_meta["content"].strip() if team_name_meta and team_name_meta.has_attr("content") else None
            nomi_squadra[ruolo_casa_ospite] = squadra_giocatore

        squadra_casa = nomi_squadra.get("home")
        squadra_ospite = nomi_squadra.get("away")

        for team_id, ruolo_casa_ospite in squadre_ids:
            if not team_id:
                continue
            team_table = soup.select_one(f"#team-{team_id}")
            if not team_table:
                continue

            squadra_giocatore = nomi_squadra.get(ruolo_casa_ospite)

            grade_table = team_table.select_one("table.grades-table")
            if not grade_table:
                continue

            fonte_headers = estrai_fonti_voto(grade_table)

            for tr in grade_table.select("tbody > tr"):
                name_el = tr.select_one("a.player-name > span")
                nome_giocatore = name_el.get_text(strip=True) if name_el else None
                if not nome_giocatore:
                    continue  # riga malformata senza nome: scartata

                role_el = tr.select_one("span.role")
                ruolo_raw = role_el.get("data-value") if role_el else None
                ruolo = RUOLO_MAP.get((ruolo_raw or "").strip().lower(), ruolo_raw)

                player_link = tr.select_one("a.player-name")
                player_id = estrai_player_id(player_link["href"]) if player_link and player_link.has_attr("href") else None

                ammonizione = False
                espulsione = False

                grade_spans = tr.select("span.player-grade")
                fantagrade_spans = tr.select("span.player-fanta-grade")

                bonus_map = {}
                for bonus_span in tr.select("div.group span.player-bonus"):
                    title = bonus_span.get("title", "").strip()
                    campo = BONUS_TITLE_MAP.get(title)
                    val = parse_number(bonus_span.get("data-value") or bonus_span.get_text())
                    if campo:
                        bonus_map[campo] = val if val is not None else bonus_map.get(campo, 0)

                for gs in grade_spans:
                    classes = gs.get("class", [])
                    if "yellow-card" in classes:
                        ammonizione = True
                    if "red-card" in classes:
                        espulsione = True

                n_fonti = max(len(grade_spans), len(fantagrade_spans), 1)
                for i in range(n_fonti):
                    voto_raw = grade_spans[i].get("data-value") if i < len(grade_spans) else None
                    fantavoto_raw = fantagrade_spans[i].get("data-value") if i < len(fantagrade_spans) else None

                    if voto_raw is None and fantavoto_raw is None:
                        continue  # nessuna pillola voto per questa fonte: nulla da registrare

                    senza_voto = (voto_raw or "").strip() == SV_SENTINEL or (fantavoto_raw or "").strip() == SV_SENTINEL
                    voto = parse_voto(voto_raw)
                    fantavoto = parse_voto(fantavoto_raw)
                    fonte = fonte_headers[i] if i < len(fonte_headers) else f"fonte_{i+1}"

                    riga = {
                        "stagione": stagione,
                        "giornata": giornata,
                        "match_id": match_id,
                        "data": data_match,
                        "squadra_casa": squadra_casa,
                        "squadra_ospite": squadra_ospite,
                        "gol_casa": gol_casa,
                        "gol_ospite": gol_ospite,
                        "squadra_giocatore": squadra_giocatore,
                        "ruolo": ruolo,
                        "nome_giocatore": nome_giocatore,
                        "player_id": player_id,
                        "fonte_voto": fonte,
                        "voto": voto,
                        "fantavoto": fantavoto,
                        "senza_voto": senza_voto,
                        "gol_fatti": bonus_map.get("gol_fatti"),
                        "gol_subiti": bonus_map.get("gol_subiti"),
                        "autogol": bonus_map.get("autogol"),
                        "rigori_segnati": bonus_map.get("rigori_segnati"),
                        "rigori_sbagliati": bonus_map.get("rigori_sbagliati"),
                        "rigori_parati": bonus_map.get("rigori_parati"),
                        "assist": bonus_map.get("assist"),
                        "ammonizione": ammonizione,
                        "espulsione": espulsione,
                        "mvp": bool(bonus_map.get("mvp")),
                    }
                    righe.append(riga)

    return righe


def giornate_gia_salvate(csv_path):
    """Ritorna il set di giornate (int) già presenti in un CSV di stagione."""
    if not csv_path.exists():
        return set()
    giornate = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                giornate.add(int(row["giornata"]))
            except (KeyError, ValueError):
                pass
    return giornate


def append_righe_csv(csv_path, righe):
    file_esiste = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_esiste:
            writer.writeheader()
        for r in righe:
            writer.writerow(r)


def concatena_aggregato():
    with open(AGGREGATE_PATH, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        writer.writeheader()
        totale = 0
        for stagione in SEASONS:
            csv_path = DATA_DIR / f"voti_{stagione}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    writer.writerow(row)
                    totale += 1
    log.info("Aggregato finale scritto in %s (%d righe totali)", AGGREGATE_PATH, totale)
    return totale


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    riepilogo = {}

    for stagione in SEASONS:
        csv_path = DATA_DIR / f"voti_{stagione}.csv"
        gia_salvate = giornate_gia_salvate(csv_path)
        righe_stagione = len(gia_salvate)
        giornate_ok = 0
        giornate_vuote = 0

        for giornata in GIORNATE:
            if giornata in gia_salvate:
                continue

            html = scarica_giornata(stagione, giornata, session)
            if html is None:
                giornate_vuote += 1
                # se la prima giornata non è disponibile, probabilmente la stagione non è
                # accessibile (redirect a stagione corrente): interrompiamo la stagione
                if giornata == 1:
                    log.info("Stagione %s non disponibile (giornata 1 assente), salto il resto", stagione)
                    break
                time.sleep(1 + random.random() * 0.5)
                continue

            righe = parsa_giornata(html, stagione, giornata)
            if righe:
                append_righe_csv(csv_path, righe)
                righe_stagione += len(righe)
                giornate_ok += 1
                log.info("%s giornata %d: %d righe salvate", stagione, giornata, len(righe))
            else:
                giornate_vuote += 1
                log.warning("%s giornata %d: nessuna riga estratta", stagione, giornata)

            time.sleep(1 + random.random() * 0.5)

        riepilogo[stagione] = {
            "righe_totali": righe_stagione,
            "giornate_ok": giornate_ok,
            "giornate_vuote_o_saltate": giornate_vuote,
        }
        log.info("=== Stagione %s completata: %s ===", stagione, riepilogo[stagione])

    totale_aggregato = concatena_aggregato()

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n=== RIEPILOGO FINALE ===\n")
        for stagione, info in riepilogo.items():
            f.write(f"{stagione}: {info}\n")
        f.write(f"TOTALE RIGHE AGGREGATO: {totale_aggregato}\n")

    log.info("Scraping completato. Totale righe aggregato: %d", totale_aggregato)


if __name__ == "__main__":
    main()
