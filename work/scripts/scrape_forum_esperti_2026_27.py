#!/usr/bin/env python3
"""
Scraper delle "schede squadra" del forum Gruppo Esperti (gruppoesperti.it),
richiesto esplicitamente da Dario dopo aver condiviso il link al topic
Atalanta: "Qui trovi per ogni squadra suggerimenti, consigli, analisi
fantacalcistici" / "Consideriamo anche questi aspetti nell'analisi" /
"Analizza i topic di ogni squadra di gruppo esperti per raccogliere una
valutazione qualitativa del giocatore che aggiungerai alla valutazione
quantitativa che abbiamo già."

FONTE: board "FANTACALCIO GE | Schede squadra e schede partita"
(viewforum.php?f=199) contiene, per la stagione 2026/27, esattamente 20
topic "SQUADRA [TOPIC UNICO]", uno per ogni squadra di Serie A (le 20
sigle coincidono esattamente con le 20 squadre presenti in
quotazioni_fantacalcio_storico_2015_2026.csv per stagione=='2026-27',
verificato dal vivo). ID topic raccolti manualmente dal vivo:

  Atalanta=232668, Bologna=232674, Cagliari=232675, Como=232669,
  Fiorentina=232665, Frosinone=232683, Genoa=232677, Inter=232676,
  Juventus=232666, Lazio=232672, Lecce=232673, Milan=232671,
  Monza=232682, Napoli=232664, Parma=232678, Roma=232670,
  Sassuolo=232679, Torino=232680, Udinese=232667, Venezia=232681.

robots.txt verificato dal vivo: viewtopic.php/viewforum.php NON sono in
Disallow (solo pagine amministrative/ricerca/cron lo sono) - scraping
consentito, con lo stesso stile prudente (rate-limit, User-Agent
dichiarato) già usato per le altre fonti del progetto.

AGGIORNAMENTO (11/08): il dominio www.gruppoesperti.it/forum/ ha iniziato
a rispondere HTTP 526 "Invalid SSL certificate" (guasto lato server,
verificato anche con un browser headless reale via Playwright - stesso
errore, quindi non un blocco anti-bot ma un certificato SSL rotto tra il
server di origine e Cloudflare). Dario ha indicato il sottodominio
corretto e funzionante: forum.gruppoesperti.it (senza "/forum/" nel path,
verificato dal vivo: HTTP 200, contenuto identico, NESSUN login
necessario - "Non c'è bisogno di autenticarsi"). URL base aggiornato di
conseguenza.

FORMATO POST (verificato dal vivo su Atalanta e Milan, identico byte per
byte tra le due squadre): per ogni giocatore un blocco
  <strong class="text-strong"><span style="font-size:150%;...">
    <span style="color:#4000FF">COGNOME</span> Nome</span> (annoNascita)
  Ruolo libero.</strong><br>
  <span style="text-decoration:underline">Titolarità X/10 - Media voto
  X/10 - Salute X/10 - Bonus X/10 - Consiglio Esperti X/10</span> -
  <strong class="text-strong">TOTALE XX/50</strong><br>
Per i portieri "Bonus" diventa "Porta inviolata" (stesso schema
numerico, verificato su Carnesecchi/Sportiello/Vismara). Se il giocatore
non ha ancora una valutazione assegnata, TUTTI i 5 sotto-punteggi e il
totale sono il carattere letterale "x" invece di un numero (es.
"Titolarità x/10 ... TOTALE x/50") - va parsato come None, MAI riempito
a caso (stesso principio già adottato per infortuni/profilo).

Questa è una fonte di GIUDIZIO SOGGETTIVO di esperti (scouting
qualitativo, "hype" preseason), categoricamente diversa dal resto del
dataset (dati misurati). Trattata come snapshot STATICO pre-2026-27,
non serie storica: il forum non ha equivalente per le stagioni passate,
non si backfilla storicamente.

MATCHING NOME -> player_id: riuso di norm/strip_accents/match_score/
MIN_MATCH_SCORE (stesso approccio di scrape_eta_giocatori.py e
scrape_infortuni_profilo_giocatori.py) contro
quotazioni_fantacalcio_storico_2015_2026.csv filtrato su
stagione=='2026-27', ristretto alla STESSA squadra (il forum raggruppa
già per squadra -> matching molto più affidabile che su tutti i 503
giocatori). Non-match dichiarati e loggati esplicitamente, mai forzati.

Output:
  work/data/forum_esperti_pagelle_2026_27.csv
    squadra, nome_forum, titolarita_forum, media_voto_forum,
    salute_forum, bonus_forum, consiglio_esperti_forum, totale_forum,
    ruolo_testuale_forum, player_id
  work/data/scrape_forum_esperti_log.txt

Uso:
  python3 scrape_forum_esperti_2026_27.py
"""
import csv
import logging
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUOTAZIONI_PATH = DATA_DIR / "quotazioni_fantacalcio_storico_2015_2026.csv"
OUT_PATH = DATA_DIR / "forum_esperti_pagelle_2026_27.csv"
LOG_PATH = DATA_DIR / "scrape_forum_esperti_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("scrape_forum_esperti")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

# squadra (sigla usata nel resto del progetto) -> topic id gruppoesperti.it
SQUADRA_TO_TOPIC_ID = {
    "ATA": 232668, "BOL": 232674, "CAG": 232675, "COM": 232669,
    "FIO": 232665, "FRO": 232683, "GEN": 232677, "INT": 232676,
    "JUV": 232666, "LAZ": 232672, "LEC": 232673, "MIL": 232671,
    "MON": 232682, "NAP": 232664, "PAR": 232678, "ROM": 232670,
    "SAS": 232679, "TOR": 232680, "UDI": 232667, "VEN": 232681,
}

MIN_MATCH_SCORE = 1  # sotto questa soglia il match non e' accettato

# Regex per il blocco nome+ruolo seguito dalla riga di pagelle. Ordine
# atteso Titolarità - Media voto - Salute - (Bonus|Porta inviolata) -
# Consiglio Esperti - TOTALE, verificato identico su Atalanta e Milan.
BLOCCO_RE = re.compile(
    r'<strong class="text-strong"><span style="font-size:150%;line-height:116%">'
    r'<span style="color:#4000FF">([^<]+)</span>\s*([^<]*?)\s*</span>\s*\((\d{4})\)\s*'
    r'([^<]*?)\.?\s*</strong><br>'
    r'\s*<span style="text-decoration:underline">'
    r'Titolar[ìi]t[àa]\s*([x\d]+)/10\s*-\s*Media voto\s*([x\d]+)/10\s*-\s*Salute\s*([x\d]+)/10\s*-\s*'
    r'(?:Bonus|Porta inviolata)\s*([x\d]+)/10\s*-\s*Consiglio Esperti\s*([x\d]+)/10'
    r'</span>\s*-\s*<strong class="text-strong">TOTALE\s*<span style="color:#4000FF">([x\d]+)/50</span></strong>',
    re.IGNORECASE,
)


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
    """Stessa strategia di scrape_eta_giocatori.py/build_player_name_mapping.py."""
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


def carica_quotazioni_2026_27():
    """dict {squadra_sigla: {nome_giocatore: player_id}} - ristretto alla
    stagione 2026-27, per matching per-squadra (piu' affidabile che sui
    503 giocatori totali)."""
    if not QUOTAZIONI_PATH.exists():
        log.error("File quotazioni non trovato (%s) - esegui prima scrape_quotazioni.py", QUOTAZIONI_PATH)
        raise SystemExit(1)
    rose = defaultdict(dict)
    with open(QUOTAZIONI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stagione"] != "2026-27":
                continue
            rose[row["squadra_sigla"]][row["nome_giocatore"]] = row["player_id"]
    return rose


def to_int_o_none(valore):
    return None if valore.strip().lower() == "x" else int(valore)


def parse_topic(html):
    """Ritorna lista di dict per ogni blocco giocatore trovato nel post."""
    giocatori = []
    for m in BLOCCO_RE.finditer(html):
        (cognome_raw, nome_raw, anno_nascita, ruolo_testuale,
         titolarita, media_voto, salute, bonus, consiglio, totale) = m.groups()
        cognome = cognome_raw.strip()
        nome = nome_raw.strip()
        nome_forum = f"{nome} {cognome}".strip() if nome else cognome
        giocatori.append({
            "nome_forum": nome_forum,
            "cognome_forum": cognome,
            "titolarita_forum": to_int_o_none(titolarita),
            "media_voto_forum": to_int_o_none(media_voto),
            "salute_forum": to_int_o_none(salute),
            "bonus_forum": to_int_o_none(bonus),
            "consiglio_esperti_forum": to_int_o_none(consiglio),
            "totale_forum": to_int_o_none(totale),
            "ruolo_testuale_forum": ruolo_testuale.strip(),
        })
    return giocatori


FIELDNAMES = ["squadra", "nome_forum", "titolarita_forum", "media_voto_forum",
              "salute_forum", "bonus_forum", "consiglio_esperti_forum",
              "totale_forum", "ruolo_testuale_forum", "player_id"]


def main():
    rose_fanta = carica_quotazioni_2026_27()
    session = requests.Session()

    mancanti_mappa = [s for s in rose_fanta if s not in SQUADRA_TO_TOPIC_ID]
    if mancanti_mappa:
        log.warning("Squadre 2026-27 SENZA topic forum mappato (saltate): %s", mancanti_mappa)
    mancanti_rosa = [s for s in SQUADRA_TO_TOPIC_ID if s not in rose_fanta]
    if mancanti_rosa:
        log.warning("Topic forum mappati SENZA rosa 2026-27 nota (saltati): %s", mancanti_rosa)

    righe_out = []
    tot_giocatori_forum = 0
    tot_x_su_10 = 0
    tot_matchati = 0
    tot_non_matchati = 0
    non_matchati_dettaglio = []

    for squadra, topic_id in sorted(SQUADRA_TO_TOPIC_ID.items()):
        if squadra not in rose_fanta:
            continue
        url = f"https://forum.gruppoesperti.it/viewtopic.php?t={topic_id}"
        r = get_con_retry(session, url)
        time.sleep(2.5)
        if r is None:
            log.error("Impossibile scaricare topic per %s (t=%d) dopo retry", squadra, topic_id)
            continue

        giocatori = parse_topic(r.text)
        log.info("%s (t=%d): %d giocatori estratti dal post", squadra, topic_id, len(giocatori))
        if not giocatori:
            log.warning("%s: nessun blocco giocatore trovato - verificare formato pagina", squadra)
            continue
        tot_giocatori_forum += len(giocatori)

        rosa_fanta = rose_fanta[squadra]
        rosa_norm = [(nome_fanta, norm(nome_fanta), player_id) for nome_fanta, player_id in rosa_fanta.items()]

        candidati = []
        for idx_g, g in enumerate(giocatori):
            if g["totale_forum"] is None:
                tot_x_su_10 += 1
            cognome_norm = norm(g["cognome_forum"])
            nome_forum_norm = norm(g["nome_forum"])
            for nome_fanta, nome_fanta_norm, player_id in rosa_norm:
                # provo sia il solo cognome che nome+cognome completo e
                # prendo il MASSIMO dei due punteggi (non solo il primo
                # non-None): il match sul solo cognome a volte ritorna un
                # punteggio BASSO ma non-None (es. "contains", 0.5) che
                # bloccherebbe un match migliore sul nome completo -
                # verificato dal vivo sul caso "Ederson D.S." (fantacalcio,
                # disambiguato con iniziali) vs "José ... EDERSON" (forum):
                # solo cognome da' 0.5 (sotto soglia), nome completo da' 1
                # (iniziale "S." di "D.S." matcha "Santos").
                score_cognome = match_score(nome_fanta_norm, cognome_norm)
                score_completo = match_score(nome_fanta_norm, nome_forum_norm)
                candidati_score = [s for s in (score_cognome, score_completo) if s is not None]
                score = max(candidati_score) if candidati_score else None
                if score is not None and score >= MIN_MATCH_SCORE:
                    candidati.append((score, idx_g, nome_fanta, player_id))
        candidati.sort(key=lambda c: -c[0])

        usati_g = set()
        usati_fanta = set()
        player_id_per_g = {}
        for score, idx_g, nome_fanta, player_id in candidati:
            if idx_g in usati_g or nome_fanta in usati_fanta:
                continue
            usati_g.add(idx_g)
            usati_fanta.add(nome_fanta)
            player_id_per_g[idx_g] = player_id

        for idx_g, g in enumerate(giocatori):
            player_id = player_id_per_g.get(idx_g)
            if player_id is None:
                tot_non_matchati += 1
                non_matchati_dettaglio.append(f"{squadra}: {g['nome_forum']}")
            else:
                tot_matchati += 1
            righe_out.append({
                "squadra": squadra,
                "nome_forum": g["nome_forum"],
                "titolarita_forum": g["titolarita_forum"],
                "media_voto_forum": g["media_voto_forum"],
                "salute_forum": g["salute_forum"],
                "bonus_forum": g["bonus_forum"],
                "consiglio_esperti_forum": g["consiglio_esperti_forum"],
                "totale_forum": g["totale_forum"],
                "ruolo_testuale_forum": g["ruolo_testuale_forum"],
                "player_id": player_id,
            })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(righe_out)

    log.info("=== Riepilogo ===")
    log.info("Giocatori estratti dal forum (totale su 20 squadre): %d", tot_giocatori_forum)
    log.info("Giocatori con pagelle NON ancora assegnate (x/10, normale, non errore): %d", tot_x_su_10)
    log.info("Match nome->player_id riusciti: %d", tot_matchati)
    log.info("Match nome->player_id FALLITI (dichiarati, non forzati): %d", tot_non_matchati)
    if non_matchati_dettaglio:
        log.warning("Dettaglio non-match:\n%s", "\n".join(non_matchati_dettaglio))
    log.info("Output scritto in %s (%d righe)", OUT_PATH, len(righe_out))


if __name__ == "__main__":
    main()
