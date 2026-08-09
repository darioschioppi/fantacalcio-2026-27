#!/usr/bin/env python3
"""
Scraper delle quotazioni ufficiali Fantacalcio.it (prezzo d'asta in
crediti, "quotazione iniziale" e "quotazione attuale", più il FVM = Fanta
Valore di Mercato) per ogni giocatore, per tutte le stagioni storiche.

Fonte segnalata da Dario: https://www.fantacalcio.it/quotazioni-fantacalcio
Verificato dal vivo (richieste HTTP dirette, read-only) che:
  - la pagina per stagione è raggiungibile su
    https://www.fantacalcio.it/quotazioni-fantacalcio/{stagione}
    (es. "2015-16", ... "2025-26"), HTTP 200, NESSUN login richiesto;
  - i dati sono incorporati direttamente nell'HTML (tabella statica), non
    serve eseguire JavaScript né chiamare l'endpoint di download Excel
    (quello richiede login: 401 senza autenticazione, quindi non lo si usa);
  - il link al profilo di ciascun giocatore contiene l'ID numerico
    fantacalcio.it (es. ".../higuain/408" oppure ".../higuain/408/2015-16"
    per le stagioni passate) - verificato che questo ID COINCIDE con lo
    stesso player_id già usato in work/data/voti_storici_2015_2026.csv
    (spot-check: Higuaín=408 in entrambe le fonti). Permette quindi un
    JOIN DIRETTO per player_id, senza bisogno di un mapping per nome.

Si estraggono solo le colonne CLASSIC (quotazione iniziale/attuale/FVM),
perché la lega di Dario è Fantacalcio Classic (non Mantra) - confermato
nelle sessioni precedenti del progetto.

NOTA: la "quotazione attuale" (QA) si aggiorna nel corso della stagione
in base al rendimento, quindi per una stagione già conclusa QA riflette
informazione di FINE stagione (non disponibile prima che la stagione N
cominci) - va usata con cautela nel dataset season-aggregate: solo la
QUOTAZIONE INIZIALE (QI) di una stagione è genuinamente pre-stagione e
utilizzabile come feature senza leakage. QA/FVM finali sono comunque
salvati qui (dataset grezzo, nessun filtro), la scelta di quali colonne
usare come feature "safe" spetta allo script builder successivo
(build_stagione_giocatore_dataset.py).

Output:
  work/data/quotazioni_fantacalcio_storico_2015_2026.csv
  colonne: stagione, player_id, nome_giocatore, squadra_sigla,
           ruolo_classic, ruolo_mantra, quotazione_iniziale,
           quotazione_attuale, fvm
  work/data/quotazioni_scrape_log.txt

Uso:
  python3 scrape_quotazioni.py
"""
import csv
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "quotazioni_scrape_log.txt"
OUT_PATH = DATA_DIR / "quotazioni_fantacalcio_storico_2015_2026.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("scrape_quotazioni")

BASE = "https://www.fantacalcio.it/quotazioni-fantacalcio"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# stagioni storiche coperte dal resto del progetto + la corrente 2026-27
# (già pubblicata all'atto della scrittura di questo script - utile come
# quotazione iniziale pre-stagione per la stagione target corrente).
STAGIONI = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    "2025-26", "2026-27",
]

PLAYER_ID_RE = re.compile(r"/squadre/[^/]+/[^/]+/(\d+)")


def parse_price(text):
    text = text.strip()
    if text in ("-", "", "N.D."):
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def scarica_stagione(stagione, session):
    url = f"{BASE}/{stagione}"
    resp = session.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        log.warning("Stagione %s: HTTP %d, salto", stagione, resp.status_code)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    righe_html = soup.select("tr.player-row")
    if not righe_html:
        log.warning("Stagione %s: nessuna riga player-row trovata, salto", stagione)
        return []

    righe = []
    for tr in righe_html:
        link = tr.select_one("a.player-name")
        if link is None or not link.get("href"):
            continue
        m = PLAYER_ID_RE.search(link["href"])
        if not m:
            continue
        player_id = int(m.group(1))
        nome = link.get_text(strip=True)

        squadra_td = tr.select_one('td[data-col-key="sq"]')
        squadra = squadra_td.get_text(strip=True) if squadra_td else None

        ruolo_classic_span = tr.select_one("span.role[data-value]")
        ruolo_classic = ruolo_classic_span["data-value"] if ruolo_classic_span else None

        ruolo_mantra = tr.get("data-filter-role-mantra")

        qi_td = tr.select_one('td[data-col-key="c_qi"]')
        qa_td = tr.select_one('td[data-col-key="c_qa"]')
        fvm_td = tr.select_one('td[data-col-key="c_fvm"]')

        righe.append({
            "stagione": stagione,
            "player_id": player_id,
            "nome_giocatore": nome,
            "squadra_sigla": squadra,
            "ruolo_classic": ruolo_classic,
            "ruolo_mantra": ruolo_mantra,
            "quotazione_iniziale": parse_price(qi_td.get_text()) if qi_td else None,
            "quotazione_attuale": parse_price(qa_td.get_text()) if qa_td else None,
            "fvm": parse_price(fvm_td.get_text()) if fvm_td else None,
        })

    log.info("Stagione %s: %d giocatori estratti", stagione, len(righe))
    return righe


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    tutte_le_righe = []
    for stagione in STAGIONI:
        righe = scarica_stagione(stagione, session)
        tutte_le_righe.extend(righe)
        time.sleep(1.0)  # rate-limit prudente, sito pubblico senza login

    if not tutte_le_righe:
        log.error("Nessuna riga estratta in nessuna stagione. Interrompo senza scrivere output.")
        return

    fieldnames = [
        "stagione", "player_id", "nome_giocatore", "squadra_sigla",
        "ruolo_classic", "ruolo_mantra",
        "quotazione_iniziale", "quotazione_attuale", "fvm",
    ]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tutte_le_righe)

    log.info("Totale righe scritte: %d in %s", len(tutte_le_righe), OUT_PATH)

    # verifica di continuità: conteggio per stagione
    per_stagione = {}
    for r in tutte_le_righe:
        per_stagione[r["stagione"]] = per_stagione.get(r["stagione"], 0) + 1
    for s in STAGIONI:
        log.info("  %s: %d righe", s, per_stagione.get(s, 0))

    # spot-check noto: Higuain deve avere player_id 408 (verificato a mano
    # in precedenza contro voti_storici_2015_2026.csv)
    higuain = [r for r in tutte_le_righe if r["player_id"] == 408]
    if higuain:
        log.info("Spot-check OK: player_id=408 trovato (%s) in %d stagioni: %s",
                  higuain[0]["nome_giocatore"], len(higuain),
                  sorted(set(r["stagione"] for r in higuain)))
    else:
        log.warning("Spot-check FALLITO: player_id=408 (Higuain) non trovato in nessuna stagione")


if __name__ == "__main__":
    main()
