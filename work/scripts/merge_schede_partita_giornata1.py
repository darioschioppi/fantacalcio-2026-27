#!/usr/bin/env python3
"""
Integra i dati delle "schede partita" della giornata 1 (scrapati da
scrape_schede_partita_2026_27.py) nel report di valutazione esistente
(previsioni_serie_a_2026_27_con_sorprese.csv), aggiungendo un nuovo
indice di rischio/sorpresa SPECIFICO per la giornata 1, richiesto da
Dario ("Analizza tutti i thread delle partite della prima giornata per
aumentare i dati della valutazione" -> scelta esplicita "Integrare
subito nel modello/report... nuovo indice di sorpresa/rischio per la
giornata 1 simile a fuzzy_sorprese_forum.py").

DIFFERENZA rispetto a fuzzy_sorprese_forum.py (indice_sorpresa
esistente): quello confronta la previsione del modello con un giudizio
FORUM-SQUADRA generico/preseason (titolarita_forum_esperti, valido per
tutta la stagione). QUESTO nuovo indice confronta invece la previsione
del modello con l'informazione SPECIFICA DELLA GIORNATA 1 (ballottaggio
percentuale reale della settimana, indisponibilita' dichiarata per
QUESTA partita) - un segnale molto piu' fresco e mirato, disponibile
solo per le 20 squadre della giornata 1 e solo per i giocatori
effettivamente menzionati nella scheda partita.

INPUT FUZZY (stesso sistema Mamdani di fuzzy_sorprese_forum.py, RIUSATO
via import - costruisci_sistema_fuzzy() accetta genericamente un
"pred_titolare" 0-35 e un "forum_titolarita" 0-10, la seconda variabile
adattata qui a "titolarita_giornata1" sulla stessa scala 0-10):
  1. pred_presenze_titolare_previste (0-35, da previsioni_con_sorprese)
  2. titolarita_giornata1 (0-10), derivata SOLO da segnali espliciti
     della scheda partita:
       - indisponibile=True (dichiarato per questa giornata) -> 0
       - ballottaggio_pct presente -> ballottaggio_pct/10
     NESSUN altro caso viene dedotto: un giocatore matchato ma SENZA
     ballottaggio ne' indisponibilita' dichiarata (probabile titolare
     "pacifico", il forum non lo segnala perche' scontato) NON riceve un
     valore inventato (es. 10) - verrebbe fabbricata un'informazione che
     il forum non ha dato esplicitamente. Stesso principio anti-
     invenzione-dati di fuzzy_sorprese_forum.py, qui applicato in modo
     ANCORA PIU' stringente perche' la fonte (ballottaggio/indisponibili)
     e' per natura un elenco di ECCEZIONI, non una lista esaustiva.

OUTPUT: indice_rischio_giornata1 (-10..+10, stessa scala/semantica di
indice_sorpresa) + categoria_rischio_giornata1. Righe senza input
disponibile (giocatore non nella giornata 1, o matchato ma senza
ballottaggio/indisponibilita' dichiarati) -> entrambe le colonne vuote.

JOIN: previsioni_con_sorprese.csv ha (squadra, nome) ma NON player_id;
schede_partita_giornata1_2026_27.csv ha player_id. Bridge tramite
quotazioni_fantacalcio_storico_2015_2026.csv filtrato 2026-27
((squadra_sigla, nome_giocatore) -> player_id), la stessa fonte da cui
"nome" in previsioni_con_sorprese e' originariamente popolato
(predict_serie_a_2026_27.py usa agg_n1['nome_giocatore'] della stessa
tabella) - quindi il match per (squadra, nome) e' by design 1:1 esatto,
NON serve fuzzy matching qui (a differenza degli scraper del forum, che
matchano nomi scritti liberamente da terzi).

Output:
  work/data/previsioni_serie_a_2026_27_giornata1.csv (tutte le colonne
    di previsioni_serie_a_2026_27_con_sorprese.csv + colonne giornata1_*
    + indice_rischio_giornata1 + categoria_rischio_giornata1)
  work/data/merge_schede_partita_giornata1_log.txt

Uso:
  python3 merge_schede_partita_giornata1.py
"""
import csv
import logging
import os
from pathlib import Path

from fuzzy_sorprese_forum import costruisci_sistema_fuzzy

DATA_DIR = Path(os.environ.get("FANTACALCIO_DATA_DIR") or (Path(__file__).resolve().parent.parent / "data"))
PREVISIONI_PATH = DATA_DIR / "previsioni_serie_a_2026_27_con_sorprese.csv"
QUOTAZIONI_PATH = DATA_DIR / "quotazioni_fantacalcio_storico_2015_2026.csv"
SCHEDE_PATH = DATA_DIR / "schede_partita_giornata1_2026_27.csv"
OUT_PATH = DATA_DIR / "previsioni_serie_a_2026_27_giornata1.csv"
LOG_PATH = DATA_DIR / "merge_schede_partita_giornata1_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("merge_schede_partita_giornata1")

NUOVE_COLONNE = [
    "giornata1_squadra_casa", "giornata1_squadra_ospite", "giornata1_data_ora",
    "titolare_previsto_giornata1", "ballottaggio_pct_giornata1",
    "indisponibile_giornata1", "motivo_indisponibilita_giornata1",
    "voto_consigliato_giornata1", "commento_voto_giornata1",
    "rigorista_giornata1", "tira_punizioni_giornata1", "tira_angoli_giornata1",
    "indice_rischio_giornata1", "categoria_rischio_giornata1",
]


def carica_player_id_map():
    """dict {(squadra, nome): player_id} - stagione 2026-27, stessa fonte
    da cui deriva 'nome' in previsioni_con_sorprese.csv (join esatto per
    costruzione, non serve fuzzy matching)."""
    mappa = {}
    with open(QUOTAZIONI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stagione"] != "2026-27":
                continue
            mappa[(row["squadra_sigla"], row["nome_giocatore"])] = row["player_id"]
    return mappa


def carica_schede_per_player_id():
    """dict {player_id: riga} da schede_partita_giornata1_2026_27.csv,
    solo righe con player_id non-None (non-match gia' esclusi qui)."""
    schede = {}
    with open(SCHEDE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["player_id"]:
                schede[row["player_id"]] = row
    return schede


def to_bool_o_none(valore):
    if valore in (None, ""):
        return None
    return valore == "True"


def categoria_rischio_da_indice(indice):
    if indice is None:
        return None
    if indice >= 3.0:
        return "OCCASIONE GIORNATA 1 (piu' titolare del previsto dal modello)"
    if indice <= -3.0:
        return "RISCHIO GIORNATA 1 (ballottaggio/indisponibilita' non previsti dal modello)"
    return "coerente (modello e scheda partita d'accordo)"


def main():
    if not PREVISIONI_PATH.exists():
        log.error("File previsioni non trovato (%s) - esegui prima fuzzy_sorprese_forum.py", PREVISIONI_PATH)
        raise SystemExit(1)
    if not SCHEDE_PATH.exists():
        log.error("File schede partita non trovato (%s) - esegui prima scrape_schede_partita_2026_27.py", SCHEDE_PATH)
        raise SystemExit(1)

    player_id_map = carica_player_id_map()
    schede = carica_schede_per_player_id()
    sim = costruisci_sistema_fuzzy()

    with open(PREVISIONI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        righe = list(reader)

    n_giornata1 = 0
    n_senza_giornata1 = 0
    n_con_indice = 0
    n_senza_indice_ma_in_giornata1 = 0
    n_rischio = 0
    n_occasione = 0
    dettaglio = []

    for r in righe:
        for c in NUOVE_COLONNE:
            r[c] = ""

        player_id = player_id_map.get((r["squadra"], r["nome"]))
        scheda = schede.get(player_id) if player_id else None
        if scheda is None:
            n_senza_giornata1 += 1
            continue

        n_giornata1 += 1
        r["giornata1_squadra_casa"] = scheda["squadra_casa"]
        r["giornata1_squadra_ospite"] = scheda["squadra_ospite"]
        r["giornata1_data_ora"] = scheda["data_ora"]
        r["titolare_previsto_giornata1"] = scheda["titolare_previsto"]
        r["ballottaggio_pct_giornata1"] = scheda["ballottaggio_pct"]
        r["indisponibile_giornata1"] = scheda["indisponibile"]
        r["motivo_indisponibilita_giornata1"] = scheda["motivo_indisponibilita"]
        r["voto_consigliato_giornata1"] = scheda["voto_consigliato"]
        r["commento_voto_giornata1"] = scheda["commento_voto"]
        r["rigorista_giornata1"] = scheda["rigorista"]
        r["tira_punizioni_giornata1"] = scheda["tira_punizioni"]
        r["tira_angoli_giornata1"] = scheda["tira_angoli"]

        indisponibile = to_bool_o_none(scheda["indisponibile"])
        ballottaggio_pct = float(scheda["ballottaggio_pct"]) if scheda["ballottaggio_pct"] else None
        pred_titolare_raw = r.get("pred_presenze_titolare_previste")

        titolarita_giornata1 = None
        if indisponibile:
            titolarita_giornata1 = 0.0
        elif ballottaggio_pct is not None:
            titolarita_giornata1 = ballottaggio_pct / 10.0

        if titolarita_giornata1 is None or not pred_titolare_raw:
            n_senza_indice_ma_in_giornata1 += 1
            continue

        pred_val = min(max(float(pred_titolare_raw), 0.0), 34.99)
        sim.input["pred_titolare"] = pred_val
        sim.input["forum_titolarita"] = min(max(titolarita_giornata1, 0.0), 10.0)
        sim.compute()
        indice = float(sim.output["sorpresa"])

        r["indice_rischio_giornata1"] = round(indice, 2)
        r["categoria_rischio_giornata1"] = categoria_rischio_da_indice(indice)
        n_con_indice += 1

        if indice >= 3.0:
            n_occasione += 1
            dettaglio.append((indice, r["squadra"], r["nome"], pred_val, titolarita_giornata1, "OCCASIONE"))
        elif indice <= -3.0:
            n_rischio += 1
            dettaglio.append((indice, r["squadra"], r["nome"], pred_val, titolarita_giornata1, "RISCHIO"))

    if not righe:
        log.warning("Nessuna riga da scrivere: input vuoto, nessun CSV di output generato.")
        return

    fieldnames = list(righe[0].keys())
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(righe)

    log.info("=== Riepilogo ===")
    log.info("Giocatori totali nel report: %d", len(righe))
    log.info("Giocatori con dati giornata1 (matchati sulla scheda partita): %d", n_giornata1)
    log.info("Giocatori SENZA dati giornata1 (non nella prima giornata o non matchati): %d", n_senza_giornata1)
    log.info("Di cui con dati giornata1 ma senza segnale per l'indice (nessun ballottaggio/indisponibilita' dichiarati): %d", n_senza_indice_ma_in_giornata1)
    log.info("Indice di rischio calcolato per: %d giocatori", n_con_indice)
    log.info("OCCASIONI giornata1 (indice>=3): %d", n_occasione)
    log.info("RISCHI giornata1 (indice<=-3): %d", n_rischio)
    log.info("Output scritto in %s", OUT_PATH)
    log.info("")
    log.info("=== Dettaglio occasioni/rischi (ordinate per intensita' indice) ===")
    dettaglio.sort(key=lambda x: -abs(x[0]))
    for indice, squadra, nome, pred_val, titol_g1, tipo in dettaglio:
        log.info("  [%s] %-6s %-25s pred_titolare=%.1f titolarita_g1=%.1f indice=%.2f",
                  tipo, squadra, nome, pred_val, titol_g1, indice)


if __name__ == "__main__":
    main()
