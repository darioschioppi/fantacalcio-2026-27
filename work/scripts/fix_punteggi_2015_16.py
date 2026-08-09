#!/usr/bin/env python3
"""
Corregge il bug scoperto durante la validazione della classifica dinamica:
per le giornate 1-3 della stagione 2015-16 (+1 caso isolato in 2016-17
giornata 2), le colonne gol_casa/gol_ospite in voti_storici_2015_2026.csv
(e nel corrispondente voti_<stagione>.csv) contengono un punteggio
placeholder/stantio invece del punteggio reale (es. Juventus-Udinese
mostrato come "4-2" quando il risultato reale fu 0-1). Causa: la pagina
fantacalcio.it per queste giornate storiche più vecchie mostra un
placeholder in span.score-home/score-away invece del punteggio reale.

I voti/bonus INDIVIDUALI dei giocatori (voto, fantavoto, gol_fatti,
assist, ecc.) NON sono contaminati - verificato separatamente - quindi
questo script tocca SOLO le colonne gol_casa/gol_ospite, riscrivendo i
CSV di stagione e l'aggregato con i valori corretti presi da Understat
(fonte già scaricata e verificata in work/data/understat_shots_*.csv).

Strategia di matching: per ogni match_id nelle stagioni coinvolte, trova
la partita Understat corrispondente per (data ± 1 giorno, squadra casa
riconosciuta per prefisso nome - stesso approccio già usato altrove in
questo progetto per gestire le differenze di naming tra fonti).

Uso:
  python3 fix_punteggi_2015_16.py
"""
import csv
import logging
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOTI_STORICO_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
LOG_PATH = DATA_DIR / "fix_punteggi_log.txt"

# Stagioni in cui è stato riscontrato il bug (vedi indagine): tutte le
# altre stagioni sono state verificate corrette e non vengono toccate.
STAGIONI_DA_CORREGGERE = ["2015-16", "2016-17"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("fix_punteggi")


def load_understat_by_date(stagione):
    path = DATA_DIR / f"understat_shots_{stagione}.csv"
    und_by_date = {}
    seen = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["match_id"]
            if key in seen:
                continue
            seen.add(key)
            d = row["date"][:10]
            und_by_date.setdefault(d, []).append(
                (row["h_team"], row["h_goals"], row["a_goals"], row["a_team"])
            )
    return und_by_date


def find_score(und_by_date, data_str, squadra_casa):
    y, m, d = (int(x) for x in data_str.split("-"))
    base = date(y, m, d)
    candidates = list(und_by_date.get(data_str, []))
    for delta in (1, -1):
        alt = (base + timedelta(days=delta)).isoformat()
        candidates += und_by_date.get(alt, [])
    for ht, hg, ag, at in candidates:
        if squadra_casa[:4].lower() in ht.lower() or ht[:4].lower() in squadra_casa.lower():
            return int(hg), int(ag)
    return None


def costruisci_correzioni():
    """Ritorna dict {(stagione, match_id): (gol_casa_corretto, gol_ospite_corretto)}"""
    correzioni = {}
    n_verificate = 0
    n_corrette = 0
    n_non_trovate = 0

    for stagione in STAGIONI_DA_CORREGGERE:
        und_by_date = load_understat_by_date(stagione)
        csv_path = DATA_DIR / f"voti_{stagione}.csv"
        partite_viste = set()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                match_id = row["match_id"]
                if match_id in partite_viste:
                    continue
                partite_viste.add(match_id)
                n_verificate += 1

                gol_casa_attuale = row["gol_casa"]
                gol_ospite_attuale = row["gol_ospite"]
                score = find_score(und_by_date, row["data"], row["squadra_casa"])
                if score is None:
                    n_non_trovate += 1
                    continue
                gc_corretto, go_corretto = score
                try:
                    gc_attuale_num = int(float(gol_casa_attuale))
                    go_attuale_num = int(float(gol_ospite_attuale))
                except (ValueError, TypeError):
                    gc_attuale_num = go_attuale_num = None

                if gc_attuale_num != gc_corretto or go_attuale_num != go_corretto:
                    correzioni[(stagione, match_id)] = (gc_corretto, go_corretto)
                    n_corrette += 1
                    log.info(
                        "%s match_id=%s %s: %s-%s -> %s-%s",
                        stagione, match_id, row["squadra_casa"],
                        gol_casa_attuale, gol_ospite_attuale, gc_corretto, go_corretto,
                    )

    log.info("Partite verificate: %d, corrette: %d, non trovate in Understat: %d",
              n_verificate, n_corrette, n_non_trovate)
    return correzioni


def applica_correzioni_file(csv_path, correzioni, stagione):
    """Riscrive un CSV applicando le correzioni per match_id, preservando
    tutte le altre colonne inalterate."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_righe_modificate = 0
    for row in rows:
        key = (stagione, row["match_id"])
        if key in correzioni:
            gc, go = correzioni[key]
            row["gol_casa"] = str(float(gc))
            row["gol_ospite"] = str(float(go))
            n_righe_modificate += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return n_righe_modificate


def main():
    correzioni = costruisci_correzioni()
    log.info("Totale correzioni da applicare: %d", len(correzioni))

    for stagione in STAGIONI_DA_CORREGGERE:
        csv_path = DATA_DIR / f"voti_{stagione}.csv"
        n = applica_correzioni_file(csv_path, correzioni, stagione)
        log.info("%s: %d righe modificate in %s", stagione, n, csv_path.name)

    n_agg = applica_correzioni_file(
        VOTI_STORICO_PATH,
        {k: v for k, v in correzioni.items()},
        None,  # placeholder, sovrascritto sotto per gestire multi-stagione
    ) if False else None

    # L'aggregato contiene righe di TUTTE le stagioni: applichiamo le
    # correzioni per ciascuna stagione coinvolta usando lo stesso match_id.
    with open(VOTI_STORICO_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_righe_modificate = 0
    for row in rows:
        key = (row["stagione"], row["match_id"])
        if key in correzioni:
            gc, go = correzioni[key]
            row["gol_casa"] = str(float(gc))
            row["gol_ospite"] = str(float(go))
            n_righe_modificate += 1

    with open(VOTI_STORICO_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log.info("Aggregato: %d righe modificate in %s", n_righe_modificate, VOTI_STORICO_PATH.name)
    log.info("Correzione completata.")


if __name__ == "__main__":
    main()
