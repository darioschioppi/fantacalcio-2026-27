#!/usr/bin/env python3
"""
Scraper dell'età dei giocatori per stagione, richiesta esplicitamente da
Dario per arricchire il modello di rendimento stagionale ("Bisogna trovare
il modo da considerare più fattori" tra cui l'età, oltre a storico esteso
e fattore Champions League).

NESSUNA fonte già integrata nel progetto (fantacalcio.it, Understat, Lega
Serie A) riporta data di nascita/età. Fonte nuova verificata dal vivo,
read-only, nessun login richiesto: **Transfermarkt** (transfermarkt.it),
pagine rosa per squadra/stagione, es.
  https://www.transfermarkt.it/como-1907/kader/verein/1047/saison_id/2025/plus/1
(HTTP 200, tabella con nome, ruolo, data di nascita completa "gg/mm/aaaa
(età)" per ogni giocatore della rosa DI QUELLA STAGIONE - `saison_id` è
l'anno di INIZIO della stagione, es. 2025 per la stagione 2025-26).

MAPPATURA SQUADRA -> ID CLUB TRANSFERMARKT: l'ID numerico del club
("verein/ID") non è deducibile dal nome, e la ricerca per nome può dare
falsi positivi omonimi (es. "Como" trova sia il vero Como 1907 = verein
1047 SIA un club olandese omonimo = verein 1090 - scoperto ed evitato
scegliendo sempre il primo risultato pertinente della sezione "Risultati
società" con lega Serie A/B/C esplicita, verificato a mano per tutte le
35 squadre comparse nel nostro storico 2015-16..2025-26). Mappa
`SQUADRA_TO_VEREIN_ID` sotto, verificata dal vivo una per una.

MATCHING NOME -> player_id fantacalcio.it: NESSUN ID condiviso tra
Transfermarkt e fantacalcio.it (a differenza delle quotazioni, dove il
link profilo embedava lo stesso player_id). Serve fuzzy-matching per nome,
stessa STRATEGIA (non stesso codice, adattata) di
build_player_name_mapping.py: normalizzazione accenti/minuscolo/punteggiatura
+ punteggio di match su cognome/token/contains, ma qui applicato per
(squadra, stagione) con liste piccole (~25-45 giocatori Transfermarkt vs
tutti i player_id fantacalcio.it che hanno giocato in quella squadra in
quella stagione - insieme comunque piccolo, quindi matching più affidabile
che nel caso Understat che copriva 20 squadre * 380 partite/stagione).
Match NON forzato: se nessun candidato supera una soglia minima di
punteggio, il giocatore Transfermarkt resta non associato e viene
loggato esplicitamente (mai un match a caso).

Età calcolata al 1 agosto dell'anno di INIZIO della stagione (convenzione:
inizio campionato Serie A).

AGGIORNAMENTO (richiesta Dario: infortuni + profilo fisico da
Transfermarkt): la pagina rosa gia' scaricata qui contiene anche l'href
del link nome giocatore (es. "/jean-butez/profil/spieler/290537"), da cui
si estrae GRATIS l'ID Transfermarkt del giocatore (`player_tm_id`), senza
bisogno di un secondo giro di fuzzy-matching. Questo ID viene poi usato
da scrape_infortuni_profilo_giocatori.py per scaricare le pagine
infortuni/profilo per-giocatore.

Output:
  work/data/eta_giocatori_storico_2015_2026.csv
    colonne: player_id, stagione, eta_al_1_agosto, nome_transfermarkt,
             squadra, match_type, player_tm_id
  work/data/eta_giocatori_scrape_log.txt (copertura % per stagione,
    elenco giocatori Transfermarkt non matchati, copertura player_tm_id)

Uso:
  python3 scrape_eta_giocatori.py
"""
import csv
import logging
import re
import time
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
OUT_PATH = DATA_DIR / "eta_giocatori_storico_2015_2026.csv"
LOG_PATH = DATA_DIR / "eta_giocatori_scrape_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("scrape_eta_giocatori")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

STAGIONI_STORICHE = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                      "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
                      "2025-26", "2026-27"]

# anno di inizio stagione (saison_id Transfermarkt = questo stesso anno)
def anno_inizio(stagione):
    return int(stagione.split("-")[0])


# Mappa squadra (nome usato in voti_storici_2015_2026.csv) -> ID club
# Transfermarkt, verificata dal vivo una per una (HTTP GET diretto sulla
# pagina squadra o sulla ricerca "schnellsuche", scegliendo sempre il primo
# risultato pertinente Serie A/B/C, MAI il primo risultato assoluto - vedi
# nota Como sopra). "slug" e' lo slug URL usato da Transfermarkt (serve per
# costruire l'URL leggibile, anche se in realta' l'ID numerico basta a
# risolvere la pagina indipendentemente dallo slug).
SQUADRA_TO_VEREIN_ID = {
    "Atalanta": 800,
    "Benevento": 4171,
    "Bologna": 1025,
    "Brescia": 132806,  # Union Brescia (nuovo club dopo fallimento 2023) - copertura limitata per 2019-20
    "Cagliari": 1390,
    "Carpi": 4102,
    "Chievo": 862,
    "Como": 1047,
    "Cremonese": 2239,
    "Crotone": 4083,
    "Empoli": 749,
    "Fiorentina": 430,
    "Frosinone": 8970,
    "Genoa": 252,
    "Inter": 46,
    "Juventus": 506,
    "Lazio": 398,
    "Lecce": 1005,
    "Milan": 5,
    "Monza": 2919,
    "Napoli": 6195,
    "Palermo": 458,
    "Parma": 130,
    "Pescara": 2921,
    "Pisa": 4172,
    "Roma": 12,
    "SPAL": 2722,
    "Salernitana": 380,
    "Sampdoria": 1038,
    "Sassuolo": 6574,
    "Spezia": 3522,
    "Torino": 416,
    "Udinese": 410,
    "Venezia": 607,
    "Verona": 276,
}

MIN_MATCH_SCORE = 1  # sotto questa soglia il match non e' accettato


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s):
    if not s:
        return ""
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_score(nome_breve_norm, nome_completo_norm):
    """Stessa strategia di build_player_name_mapping.py: punteggio di
    match tra un nome fantacalcio.it (tipicamente cognome, talvolta con
    iniziale) e un nome Transfermarkt (nome completo)."""
    tokens = nome_completo_norm.split()
    if nome_breve_norm == nome_completo_norm:
        return 3
    if tokens and nome_breve_norm == tokens[-1]:
        return 2
    if nome_breve_norm in tokens:
        return 2
    breve_tokens = nome_breve_norm.split()
    if len(breve_tokens) >= 2 and breve_tokens[0] in tokens:
        iniziale = breve_tokens[-1].rstrip(".")
        if any(t.startswith(iniziale) for t in tokens if t != breve_tokens[0]):
            return 1
    if nome_completo_norm.endswith(nome_breve_norm) or nome_breve_norm.endswith(nome_completo_norm):
        return 1
    if nome_breve_norm in nome_completo_norm or nome_completo_norm in nome_breve_norm:
        return 0.5
    return None


def carica_rose_fantacalcio():
    """dict {(squadra, stagione): {nome_giocatore: player_id}} - dalle
    presenze in voti_storici_2015_2026.csv (qualsiasi presenza, anche
    senza voto, basta per sapere chi era in rosa quella stagione)."""
    rose = defaultdict(dict)
    with open(VOTI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["squadra_giocatore"], row["stagione"])
            rose[key][row["nome_giocatore"]] = row["player_id"]
    return rose


def get_con_retry(session, url, tentativi=4, timeout=40):
    for i in range(tentativi):
        try:
            r = session.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            log.warning("HTTP %d per %s (tentativo %d/%d)", r.status_code, url, i + 1, tentativi)
        except requests.RequestException as e:
            log.warning("Errore rete per %s: %s (tentativo %d/%d)", url, e, i + 1, tentativi)
        time.sleep(3 * (i + 1))
    return None


def estrai_player_tm_id(href):
    """href tipico: /jean-butez/profil/spieler/290537 -> 290537 (int).
    None se il pattern non matcha (dovrebbe essere sempre presente)."""
    if not href:
        return None
    m = re.search(r"/spieler/(\d+)", href)
    return int(m.group(1)) if m else None


def scarica_rosa_transfermarkt(session, verein_id, saison_id):
    url = f"https://www.transfermarkt.it/-/kader/verein/{verein_id}/saison_id/{saison_id}/plus/1"
    r = get_con_retry(session, url)
    if r is None:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", class_="items")
    if table is None:
        return []
    righe = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 3:
            continue
        nome_ruolo_td = tds[1]
        a = nome_ruolo_td.find("a")
        nome = a.get_text(strip=True) if a else nome_ruolo_td.get_text(" ", strip=True)
        # ID giocatore Transfermarkt (richiesto da Dario per arricchire con
        # infortuni/profilo fisico via scrape_infortuni_profilo_giocatori.py):
        # estratto GRATIS dallo stesso href già presente in questa tabella,
        # nessuna richiesta HTTP aggiuntiva necessaria qui.
        player_tm_id = estrai_player_tm_id(a.get("href")) if a else None
        data_nascita_raw = tds[2].get_text(strip=True)
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", data_nascita_raw)
        if not m:
            continue  # riga senza data di nascita nota (es. società figlie senza dettaglio)
        gg, mm, aaaa = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            data_nascita = date(aaaa, mm, gg)
        except ValueError:
            continue
        righe.append({"nome": nome, "data_nascita": data_nascita, "player_tm_id": player_tm_id})
    return righe


def eta_al_1_agosto(data_nascita, anno_stagione):
    riferimento = date(anno_stagione, 8, 1)
    eta = riferimento.year - data_nascita.year - (
        (riferimento.month, riferimento.day) < (data_nascita.month, data_nascita.day)
    )
    return eta


def main():
    rose_fanta = carica_rose_fantacalcio()
    session = requests.Session()

    righe_out = []
    non_matchati = []
    per_stagione_stat = defaultdict(lambda: {"tot_fanta": 0, "matchati": 0})

    squadre_nello_storico = sorted(set(k[0] for k in rose_fanta.keys()))
    log.info("Squadre nello storico voti: %d", len(squadre_nello_storico))
    mancanti_mappa = [s for s in squadre_nello_storico if s not in SQUADRA_TO_VEREIN_ID]
    if mancanti_mappa:
        log.warning("Squadre SENZA mappatura verein_id (saltate): %s", mancanti_mappa)

    for stagione in STAGIONI_STORICHE:
        anno = anno_inizio(stagione)
        for squadra, verein_id in SQUADRA_TO_VEREIN_ID.items():
            key = (squadra, stagione)
            rosa_fanta = rose_fanta.get(key)
            if not rosa_fanta:
                continue  # la squadra non era in Serie A/B quella stagione (fuori dal nostro storico voti)

            per_stagione_stat[stagione]["tot_fanta"] += len(rosa_fanta)

            rosa_tm = scarica_rosa_transfermarkt(session, verein_id, anno)
            time.sleep(1.2)  # rate-limit prudente
            if rosa_tm is None:
                log.error("Impossibile scaricare rosa Transfermarkt per %s %s (verein %d) dopo retry", squadra, stagione, verein_id)
                continue
            if not rosa_tm:
                log.warning("Rosa Transfermarkt vuota per %s %s (verein %d)", squadra, stagione, verein_id)
                continue

            rosa_tm_norm = [(r["nome"], norm(r["nome"]), r["data_nascita"], r["player_tm_id"]) for r in rosa_tm]
            usati_tm = set()

            candidati = []
            for nome_fanta, player_id in rosa_fanta.items():
                nf_norm = norm(nome_fanta)
                for idx, (nome_tm, nome_tm_norm, dob, tm_id) in enumerate(rosa_tm_norm):
                    score = match_score(nf_norm, nome_tm_norm)
                    if score is not None and score >= MIN_MATCH_SCORE:
                        candidati.append((score, nome_fanta, player_id, idx, nome_tm, dob, tm_id))
            candidati.sort(key=lambda c: -c[0])

            usati_fanta = set()
            for score, nome_fanta, player_id, idx, nome_tm, dob, tm_id in candidati:
                if nome_fanta in usati_fanta or idx in usati_tm:
                    continue
                usati_fanta.add(nome_fanta)
                usati_tm.add(idx)
                eta = eta_al_1_agosto(dob, anno)
                righe_out.append({
                    "player_id": player_id,
                    "stagione": stagione,
                    "eta_al_1_agosto": eta,
                    "nome_transfermarkt": nome_tm,
                    "squadra": squadra,
                    "match_type": "cognome/token" if score >= 2 else ("iniziale" if score == 1 else "contains"),
                    "player_tm_id": tm_id,
                })
                per_stagione_stat[stagione]["matchati"] += 1

            for nome_fanta in rosa_fanta:
                if nome_fanta not in usati_fanta:
                    non_matchati.append((squadra, stagione, nome_fanta))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["player_id", "stagione", "eta_al_1_agosto", "nome_transfermarkt",
                      "squadra", "match_type", "player_tm_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(righe_out)

    log.info("Righe eta' scritte: %d in %s", len(righe_out), OUT_PATH)

    log.info("=== Copertura per stagione ===")
    for s in STAGIONI_STORICHE:
        stat = per_stagione_stat.get(s)
        if stat and stat["tot_fanta"] > 0:
            log.info("  %s: %d/%d giocatori matchati (%.1f%%)",
                      s, stat["matchati"], stat["tot_fanta"], 100 * stat["matchati"] / stat["tot_fanta"])

    log.info("Giocatori fantacalcio.it NON matchati a nessun nome Transfermarkt: %d", len(non_matchati))
    for squadra, stagione, nome in non_matchati[:50]:
        log.info("  NON MATCHATO: %s | %s | %s", squadra, stagione, nome)
    if len(non_matchati) > 50:
        log.info("  ... e altri %d non matchati (troncato nel log)", len(non_matchati) - 50)

    # player_tm_id: richiesto da Dario per collegare gratis l'ID Transfermarkt
    # (necessario per scrape_infortuni_profilo_giocatori.py) senza un secondo
    # giro di fuzzy-matching - verifica quanti record hanno l'id popolato.
    con_tm_id = sum(1 for r in righe_out if r["player_tm_id"] is not None)
    log.info("Righe con player_tm_id popolato: %d/%d (%.1f%%)",
              con_tm_id, len(righe_out), 100 * con_tm_id / len(righe_out) if righe_out else 0)
    player_id_distinti_con_tm = len(set(r["player_id"] for r in righe_out if r["player_tm_id"] is not None))
    log.info("player_id fantacalcio distinti con almeno un player_tm_id noto: %d", player_id_distinti_con_tm)


if __name__ == "__main__":
    main()
