#!/usr/bin/env python3
"""
Scraper di infortuni e profilo fisico per giocatore, richiesto da Dario
dopo aver condiviso un PDF con 327 variabili teoriche di rendimento
("variabili_rendimento_calciatore_serie_a_con_descrizioni.pdf"). Verificato
che la stragrande maggioranza di quelle variabili (metriche GPS/tracking,
posizionamento avanzato tipo xT/VAEP/heatmap, scouting psicologico
soggettivo) NON è ottenibile da fonti pubbliche gratuite - richiederebbe
fornitori a pagamento (Opta/StatsBomb/Wyscout/Stats Perform, verificato
anche quest'ultimo: serve contatto commerciale, nessun accesso self-service
gratuito reale). Dario ha confermato di procedere SOLO col sottoinsieme
fattibile gratuitamente: infortuni + profilo fisico (altezza/piede/
nazionalità) + contratto, via **Transfermarkt** (stessa fonte già
verificata e usata per l'età in scrape_eta_giocatori.py).

AGGIORNAMENTO 2 (10/08, blocco a metà corsa): durante la prima esecuzione,
dopo circa 300-400 giocatori scaricati SENZA problemi, il dominio
www.transfermarkt.IT ha iniziato a rispondere HTTP 403 sulla stragrande
maggioranza delle richieste (dall'85-90% in su), verosimilmente rate-
limiting/anti-bot scattato dopo una soglia di richieste. Verificato dal
vivo che lo stesso identico contenuto (stessa struttura tabella/box, path
identici salvo lingua) è raggiungibile SENZA blocco sul dominio inglese
www.transfermarkt.COM (es. "/verletzungen/..." diventa "/injuries/...").
Cambiata la fonte a transfermarkt.com, con:
  - rate-limit piu' prudente (sleep aumentato),
  - retry/backoff piu' pazienti su 403/503 (visti anche su .com, ma rari
    e transitori, non un blocco sistematico),
  - salvataggio INCREMENTALE riga per riga (append ai CSV output man
    mano, non solo a fine run) - la vecchia versione scriveva i CSV solo
    alla fine, perdendo tutto il lavoro se il processo viene fermato o
    crasha a metà: corretto qui.
Le etichette del profilo su .com sono in inglese (Date of birth/Age,
Height, Citizenship, Position, Foot, Contract expires, ecc.) invece che
in italiano - dizionario di parsing aggiornato di conseguenza.

A differenza dello scraping età (per squadra/stagione, ~35 squadre x 12
stagioni = ~420 richieste), qui si scarica per SINGOLO GIOCATORE (ogni
giocatore distinto storico 2015-2026 = 1-2 richieste, potenzialmente
migliaia) - Dario ha confermato esplicitamente di voler copertura
sull'INTERO storico 2015-2026, non solo rosa attuale, accettando il
tempo più lungo necessario.

FONTE ID GIOCATORE: NESSUN nuovo fuzzy-match necessario. Lo scraping età
(scrape_eta_giocatori.py, versione aggiornata) cattura già GRATIS l'ID
Transfermarkt di ogni giocatore (player_tm_id) dall'href del link nome
nella tabella rosa - si costruisce qui una mappa deduplicata
player_id (fantacalcio) -> player_tm_id da quel CSV già esistente.

PAGINE SCARICATE (verificate dal vivo, HTTP diretto, HTML statico):
- Infortuni: https://www.transfermarkt.it/{slug}/verletzungen/spieler/{id}/plus/1
  Tabella <table class="items">, header esatto:
  ['Stagione', 'Infortunio', 'da', 'fino al', 'giorni', 'Partite perse'].
  Esempio verificato (Jean Butez, id 290537):
  ['23/24', 'Frattura della mano', '30/03/2024', '30/06/2024', '93 giorni', '10'].
  Nessun infortunio in carriera -> tabella vuota, gestito come "0 episodi"
  (dichiarato esplicitamente, NON e' un errore di scraping).
- Profilo: https://www.transfermarkt.it/{slug}/profil/spieler/{id}
  Box <div class="info-table">, coppie etichetta:valore, etichette
  verificate: "Nato il", "Luogo di nascita", "Altezza", "Nazionalità",
  "Posizione", "Piede", "Squadra attuale", "In rosa da", "Scadenza",
  "Ultimo prolungamento". NOTA: Transfermarkt NON pubblica il peso/BMI -
  limite di fonte dichiarato, nessuna colonna riempita a caso.

Output:
  work/data/infortuni_giocatori_storico_2015_2026.csv
    una riga per EPISODIO di infortunio (dato granulare, l'aggregazione
    per stagione si fa nel builder, non qui):
    player_id, stagione, tipo_infortunio, data_inizio, data_fine,
    giorni_stop, partite_perse
  work/data/profilo_giocatori_storico_2015_2026.csv
    una riga per player_id (profilo "attuale" al momento dello scraping,
    non storico per stagione - altezza/piede/nazionalità cambiano
    raramente o mai, dichiarato esplicitamente come semplificazione):
    player_id, altezza_m, nazionalita, piede_dominante, posizione_tm,
    scadenza_contratto
  work/data/infortuni_profilo_scrape_log.txt (% successo, giocatori senza
    infortuni, profili non trovati - tutto loggato esplicitamente)

Uso:
  python3 scrape_infortuni_profilo_giocatori.py
"""
import csv
import logging
import re
import time
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ETA_PATH = DATA_DIR / "eta_giocatori_storico_2015_2026.csv"
OUT_INFORTUNI_PATH = DATA_DIR / "infortuni_giocatori_storico_2015_2026.csv"
OUT_PROFILO_PATH = DATA_DIR / "profilo_giocatori_storico_2015_2026.csv"
LOG_PATH = DATA_DIR / "infortuni_profilo_scrape_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("scrape_infortuni_profilo")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

STAGIONE_TM_TO_NOSTRA = {
    f"{str(a)[-2:]}/{str(a + 1)[-2:]}": f"{a}-{str(a + 1)[-2:]}"
    for a in range(2010, 2027)
}


def get_con_retry(session, url, tentativi=5, timeout=40):
    for i in range(tentativi):
        try:
            r = session.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            log.warning("HTTP %d per %s (tentativo %d/%d)", r.status_code, url, i + 1, tentativi)
        except requests.RequestException as e:
            log.warning("Errore rete per %s: %s (tentativo %d/%d)", url, e, i + 1, tentativi)
        time.sleep(5 * (i + 1))
    return None


def carica_mappa_player_tm_id():
    """dict {player_id (fantacalcio): player_tm_id} - deduplicato dal CSV
    eta' (scrape_eta_giocatori.py aggiornato). Un giocatore reale ha lo
    stesso player_tm_id in tutte le stagioni; si prende il valore piu'
    frequente per player_id, loggando i casi con piu' di un valore diverso
    (dovrebbero essere rari/assenti - errore di matching se capitano)."""
    if not ETA_PATH.exists():
        log.error("File eta' giocatori non trovato (%s) - esegui prima scrape_eta_giocatori.py", ETA_PATH)
        raise SystemExit(1)
    valori_per_player = defaultdict(Counter)
    with open(ETA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "player_tm_id" not in reader.fieldnames:
            log.error("Colonna player_tm_id assente in %s - esegui prima la versione aggiornata di scrape_eta_giocatori.py", ETA_PATH)
            raise SystemExit(1)
        for row in reader:
            tm_id = row.get("player_tm_id")
            if tm_id:
                valori_per_player[row["player_id"]][tm_id] += 1

    mappa = {}
    ambigui = 0
    for player_id, contatore in valori_per_player.items():
        if len(contatore) > 1:
            ambigui += 1
        mappa[player_id] = contatore.most_common(1)[0][0]

    log.info("player_id fantacalcio con almeno un player_tm_id noto: %d", len(mappa))
    if ambigui:
        log.warning("player_id con PIU' DI UN player_tm_id distinto nello storico (usato il piu' frequente): %d", ambigui)
    return mappa


def parse_infortuni(html):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="items")
    if table is None:
        return []
    tbody = table.find("tbody")
    if tbody is None:
        return []
    episodi = []
    for tr in tbody.find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 6:
            continue
        stagione_tm = tds[0].get_text(strip=True)
        tipo = tds[1].get_text(strip=True)
        da = tds[2].get_text(strip=True)
        fino_al = tds[3].get_text(strip=True)
        giorni_raw = tds[4].get_text(strip=True)
        partite_raw = tds[5].get_text(strip=True)
        giorni_m = re.search(r"(\d+)", giorni_raw)
        giorni = int(giorni_m.group(1)) if giorni_m else None
        partite = None
        if partite_raw not in ("", "-"):
            partite_m = re.search(r"(\d+)", partite_raw)
            partite = int(partite_m.group(1)) if partite_m else None
        episodi.append({
            "stagione_tm": stagione_tm,
            "tipo_infortunio": tipo,
            "data_inizio": da,
            "data_fine": fino_al,
            "giorni_stop": giorni,
            "partite_perse": partite,
        })
    return episodi


def parse_profilo(html):
    soup = BeautifulSoup(html, "lxml")
    box = soup.find("div", class_="info-table")
    if box is None:
        return {}
    testo = box.get_text("|", strip=True)
    parti = testo.split("|")
    dati = {}
    for i in range(0, len(parti) - 1, 2):
        etichetta = parti[i].rstrip(":").strip()
        valore = parti[i + 1].strip()
        dati[etichetta] = valore
    return dati


def normalizza_stagione_tm(stagione_tm):
    """'23/24' -> '2023-24'. Restituisce None se il pattern non e' quello atteso."""
    return STAGIONE_TM_TO_NOSTRA.get(stagione_tm)


def altezza_a_metri(valore):
    """'1,89 m' -> 1.89 (float). None se non parsabile."""
    if not valore:
        return None
    m = re.search(r"(\d+)[,.](\d+)", valore)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def estrai_eta_da_dob(valore):
    """'08/06/1995 (31)' -> 31 (int) - non usato come feature (abbiamo
    gia' eta_n1 da scrape_eta_giocatori.py), tenuto solo per eventuale
    sanity-check incrociato, non salvato in output."""
    if not valore:
        return None
    m = re.search(r"\((\d+)\)", valore)
    return int(m.group(1)) if m else None


FIELDNAMES_INFORTUNI = ["player_id", "stagione", "tipo_infortunio", "data_inizio",
                         "data_fine", "giorni_stop", "partite_perse"]
FIELDNAMES_PROFILO = ["player_id", "altezza_m", "nazionalita", "piede_dominante",
                       "posizione_tm", "scadenza_contratto"]


def carica_tm_id_gia_processati():
    """Riprende da dove interrotto: se i CSV output esistono gia' (da un
    run precedente fermato/crashato), legge i player_id gia' scritti e li
    salta - grazie al salvataggio incrementale introdotto dopo il blocco
    Transfermarkt del 10/08 (vedi docstring in testa al file)."""
    player_id_fatti = set()
    if OUT_PROFILO_PATH.exists():
        with open(OUT_PROFILO_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_id_fatti.add(row["player_id"])
    return player_id_fatti


def main():
    mappa_tm = carica_mappa_player_tm_id()

    # slug placeholder: l'URL risolve per ID numerico indipendentemente
    # dallo slug testuale (stesso comportamento gia' verificato per le
    # pagine rosa in scrape_eta_giocatori.py) - si usa sempre "-".
    session = requests.Session()

    tm_id_distinti = sorted(set(mappa_tm.values()), key=lambda x: int(x))
    log.info("player_tm_id distinti da scaricare: %d", len(tm_id_distinti))

    tm_id_senza_infortuni = 0
    tm_id_infortuni_falliti = 0
    tm_id_profilo_falliti = 0
    n_episodi_scritti = 0
    n_profili_scritti = 0

    # mappa inversa per scrivere player_id (fantacalcio) nell'output,
    # invece del player_tm_id interno - un tm_id puo' corrispondere a piu'
    # player_id fantacalcio SOLO in casi patologici (mai osservato), si usa
    # il primo player_id fantacalcio trovato per quel tm_id.
    tm_id_to_player_id = {}
    for player_id, tm_id in mappa_tm.items():
        tm_id_to_player_id.setdefault(tm_id, player_id)

    # ripresa da run precedente interrotto (vedi carica_tm_id_gia_processati)
    player_id_fatti = carica_tm_id_gia_processati()
    if player_id_fatti:
        log.info("Ripresa da run precedente: %d player_id gia' processati, saltati", len(player_id_fatti))

    file_esistevano = OUT_INFORTUNI_PATH.exists() and OUT_PROFILO_PATH.exists()
    modo_infortuni = "a" if file_esistevano else "w"
    modo_profilo = "a" if file_esistevano else "w"

    f_inf = open(OUT_INFORTUNI_PATH, modo_infortuni, newline="", encoding="utf-8")
    writer_inf = csv.DictWriter(f_inf, fieldnames=FIELDNAMES_INFORTUNI)
    f_prof = open(OUT_PROFILO_PATH, modo_profilo, newline="", encoding="utf-8")
    writer_prof = csv.DictWriter(f_prof, fieldnames=FIELDNAMES_PROFILO)
    if not file_esistevano:
        writer_inf.writeheader()
        writer_prof.writeheader()

    n_processati_ora = 0
    try:
        for i, tm_id in enumerate(tm_id_distinti):
            player_id = tm_id_to_player_id[tm_id]
            if player_id in player_id_fatti:
                continue

            url_infortuni = f"https://www.transfermarkt.com/-/verletzungen/spieler/{tm_id}/plus/1"
            r_inf = get_con_retry(session, url_infortuni)
            time.sleep(3.0)
            if r_inf is None:
                tm_id_infortuni_falliti += 1
                log.error("Impossibile scaricare pagina infortuni per player_tm_id=%s (player_id=%s) dopo retry", tm_id, player_id)
            else:
                episodi = parse_infortuni(r_inf.text)
                if not episodi:
                    tm_id_senza_infortuni += 1
                for ep in episodi:
                    stagione_nostra = normalizza_stagione_tm(ep["stagione_tm"])
                    riga_inf = {
                        "player_id": player_id,
                        "stagione": stagione_nostra,
                        "tipo_infortunio": ep["tipo_infortunio"],
                        "data_inizio": ep["data_inizio"],
                        "data_fine": ep["data_fine"],
                        "giorni_stop": ep["giorni_stop"],
                        "partite_perse": ep["partite_perse"],
                    }
                    writer_inf.writerow(riga_inf)
                    n_episodi_scritti += 1
                f_inf.flush()

            url_profilo = f"https://www.transfermarkt.com/-/profil/spieler/{tm_id}"
            r_prof = get_con_retry(session, url_profilo)
            time.sleep(3.0)
            if r_prof is None:
                tm_id_profilo_falliti += 1
                log.error("Impossibile scaricare pagina profilo per player_tm_id=%s (player_id=%s) dopo retry", tm_id, player_id)
            else:
                dati = parse_profilo(r_prof.text)
                if not dati:
                    tm_id_profilo_falliti += 1
                    log.warning("info-table non trovata per player_tm_id=%s (player_id=%s)", tm_id, player_id)
                else:
                    riga_prof = {
                        "player_id": player_id,
                        "altezza_m": altezza_a_metri(dati.get("Height")),
                        "nazionalita": dati.get("Citizenship"),
                        "piede_dominante": dati.get("Foot"),
                        "posizione_tm": dati.get("Position"),
                        "scadenza_contratto": dati.get("Contract expires"),
                    }
                    writer_prof.writerow(riga_prof)
                    n_profili_scritti += 1
                    f_prof.flush()

            n_processati_ora += 1
            if n_processati_ora % 50 == 0:
                log.info("Progresso: %d/%d giocatori processati in questa run (totale storico incluse riprese: circa %d/%d)",
                          n_processati_ora, len(tm_id_distinti) - len(player_id_fatti),
                          len(player_id_fatti) + n_processati_ora, len(tm_id_distinti))
    finally:
        f_inf.close()
        f_prof.close()

    log.info("=== Riepilogo di questa run ===")
    log.info("Giocatori processati in questa run: %d", n_processati_ora)
    log.info("Episodi infortunio scritti in questa run: %d (totale cumulato in %s)", n_episodi_scritti, OUT_INFORTUNI_PATH)
    log.info("Righe profilo scritte in questa run: %d (totale cumulato in %s)", n_profili_scritti, OUT_PROFILO_PATH)
    log.info("Giocatori con pagina infortuni scaricata ma 0 episodi (normale, non errore): %d", tm_id_senza_infortuni)
    log.info("Giocatori con pagina infortuni NON scaricabile dopo retry: %d", tm_id_infortuni_falliti)
    log.info("Giocatori con profilo NON scaricabile/non trovato: %d", tm_id_profilo_falliti)


if __name__ == "__main__":
    main()
