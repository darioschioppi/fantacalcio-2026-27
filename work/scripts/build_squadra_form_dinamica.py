#!/usr/bin/env python3
"""
Estende la logica di build_classifica_dinamica.py con la "forma rolling"
di squadra: media di punti/gol fatti/gol subiti/xG for/xG against sulle
ULTIME 3 e ULTIME 5 partite PRECEDENTI (mai includendo la partita della
giornata corrente, per evitare data leakage - stesso principio già
applicato con successo in build_classifica_dinamica.py per punti/posizione
di classifica).

Questo sostituisce, come feature del modello previsionale, le colonne
squadra_total-points/goals/expectedGoals/... di build_feature_dataset.py,
che sono TOTALI DI FINE STAGIONE applicati identicamente a ogni giornata
(leakage: alla giornata 3 il modello "vedeva" già il rendimento di tutta la
stagione, incluse le partite future rispetto a quella giornata).

Fonti:
  - work/data/voti_storici_2015_2026.csv: risultati partita per partita
    (stagione, giornata, match_id, data, squadra_casa/ospite, gol_casa/ospite),
    deduplicati per match_id (stesso pattern di build_classifica_dinamica.py).
  - work/data/understat_shots_storico_2015_2026.csv: tiri con xG, usati per
    calcolare l'xG di squadra per partita (somma xG dei tiri della squadra
    in questione = xG "for"; somma xG dei tiri della squadra avversaria in
    quella partita = xG "against"). Il match_id di Understat è diverso da
    quello di fantacalcio.it, quindi il join usa (stagione, data ±1 giorno,
    nome squadra normalizzato) - stesso approccio già validato al 99%+ in
    build_feature_dataset.py/build_feature_dataset_v2.py per lo scarto
    UTC/locale e le differenze di denominazione squadra (Milan/AC Milan,
    SPAL/SPAL 2013, Parma/Parma Calcio 1913 - stessa mappa TEAM_NAME_MAP_UNDERSTAT
    già in uso).

Il rolling è calcolato SEPARATAMENTE per ogni stagione (le squadre cambiano
per promozione/retrocessione, non ha senso portare "forma" da una stagione
all'altra), ordinando le giornate in ordine numerico crescente - stessa
convenzione di build_classifica_dinamica.py (non l'ordine cronologico reale
per data, per coerenza con quello script già validato contro classifiche
ufficiali reali).

Per le prime giornate di una stagione, la finestra rolling è PARZIALE
(es. alla giornata 2 c'è solo 1 partita precedente disponibile) o assente
(giornata 1: nessuna storia, valori vuoti/NaN, non zero - zero sarebbe un
valore finto che il modello potrebbe interpretare come "pessima forma"
invece di "nessuna informazione"). Una colonna n_partite_forma indica
quante partite precedenti sono realmente disponibili in ciascuna finestra.

Output:
  work/data/squadra_form_dinamica_storico_2015_2026.csv
  colonne: stagione, giornata, squadra,
           forma3_punti_mean, forma3_gf_mean, forma3_gs_mean,
           forma3_xg_for_mean, forma3_xg_against_mean, forma3_n_partite,
           forma5_punti_mean, forma5_gf_mean, forma5_gs_mean,
           forma5_xg_for_mean, forma5_xg_against_mean, forma5_n_partite

Uso:
  python3 build_squadra_form_dinamica.py
"""
import csv
import logging
from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
SHOTS_PATH = DATA_DIR / "understat_shots_storico_2015_2026.csv"
OUT_PATH = DATA_DIR / "squadra_form_dinamica_storico_2015_2026.csv"
LOG_PATH = DATA_DIR / "squadra_form_dinamica_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("squadra_form_dinamica")

# stessa mappa di build_feature_dataset.py (nomi squadra fantacalcio.it -> Understat)
TEAM_NAME_MAP_UNDERSTAT = {
    "SPAL": "SPAL 2013",
    "Milan": "AC Milan",
    "Parma": "Parma Calcio 1913",
}

FINESTRE = (3, 5)


def norm_team_understat(name):
    return TEAM_NAME_MAP_UNDERSTAT.get(name, name)


def carica_partite_univoche():
    """dict {stagione: list of dict con giornata, match_id, data, squadra_casa,
    squadra_ospite, gol_casa, gol_ospite}, deduplicate per (stagione, match_id)."""
    partite_per_stagione = defaultdict(list)
    visti = set()
    with open(VOTI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["stagione"], row["match_id"])
            if key in visti:
                continue
            visti.add(key)
            try:
                giornata = int(row["giornata"])
                gol_casa = float(row["gol_casa"]) if row["gol_casa"] not in (None, "") else None
                gol_ospite = float(row["gol_ospite"]) if row["gol_ospite"] not in (None, "") else None
            except (ValueError, TypeError):
                continue
            if gol_casa is None or gol_ospite is None:
                continue
            partite_per_stagione[row["stagione"]].append({
                "giornata": giornata,
                "match_id": row["match_id"],
                "data": row["data"],
                "squadra_casa": row["squadra_casa"],
                "squadra_ospite": row["squadra_ospite"],
                "gol_casa": gol_casa,
                "gol_ospite": gol_ospite,
            })
    return partite_per_stagione


def carica_xg_partite():
    """dict {(stagione, data, h_team, a_team): (xg_home, xg_away)}
    aggregando la somma di xG dei tiri per squadra/partita da understat_shots."""
    xg_sum = defaultdict(lambda: [0.0, 0.0])  # [xg_home, xg_away]
    with open(SHOTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = (row["date"] or "")[:10]
            key = (row["stagione"], data, row["h_team"], row["a_team"])
            try:
                xg = float(row["xG"]) if row["xG"] not in (None, "") else 0.0
            except ValueError:
                xg = 0.0
            if row["h_a"] == "h":
                xg_sum[key][0] += xg
            elif row["h_a"] == "a":
                xg_sum[key][1] += xg
    log.info("Aggregate xG di squadra per %d combinazioni (stagione, data, casa, ospite)", len(xg_sum))
    return xg_sum


def trova_xg(xg_sum, stagione, data_str, team_casa_u, team_ospite_u):
    key = (stagione, data_str, team_casa_u, team_ospite_u)
    if key in xg_sum:
        return xg_sum[key]
    try:
        y, m, d = (int(x) for x in data_str.split("-"))
        base = date(y, m, d)
    except (ValueError, AttributeError):
        return None
    for delta in (1, -1):
        alt_data = (base + timedelta(days=delta)).isoformat()
        alt_key = (stagione, alt_data, team_casa_u, team_ospite_u)
        if alt_key in xg_sum:
            return xg_sum[alt_key]
    return None


def calcola_forma_dinamica(partite_per_stagione, xg_sum):
    righe_out = []
    n_match_xg = 0
    n_match_totale = 0

    for stagione, partite in partite_per_stagione.items():
        partite.sort(key=lambda p: p["giornata"])

        giornate_presenti = sorted(set(p["giornata"] for p in partite))
        squadre = set()
        for p in partite:
            squadre.add(p["squadra_casa"])
            squadre.add(p["squadra_ospite"])

        # deque delle ultime 5 partite per squadra: ognuna è dict con punti, gf, gs, xg_for, xg_against
        storico = {sq: deque(maxlen=max(FINESTRE)) for sq in squadre}

        for giornata in giornate_presenti:
            # 1. Scrivi la forma PRE-partita per questa giornata (storico accumulato fino a giornata-1)
            for sq in squadre:
                riga = {"stagione": stagione, "giornata": giornata, "squadra": sq}
                for w in FINESTRE:
                    ultime = list(storico[sq])[-w:]
                    n = len(ultime)
                    riga[f"forma{w}_n_partite"] = n
                    if n == 0:
                        riga[f"forma{w}_punti_mean"] = None
                        riga[f"forma{w}_gf_mean"] = None
                        riga[f"forma{w}_gs_mean"] = None
                        riga[f"forma{w}_xg_for_mean"] = None
                        riga[f"forma{w}_xg_against_mean"] = None
                    else:
                        riga[f"forma{w}_punti_mean"] = sum(x["punti"] for x in ultime) / n
                        riga[f"forma{w}_gf_mean"] = sum(x["gf"] for x in ultime) / n
                        riga[f"forma{w}_gs_mean"] = sum(x["gs"] for x in ultime) / n
                        xg_vals_for = [x["xg_for"] for x in ultime if x["xg_for"] is not None]
                        xg_vals_against = [x["xg_against"] for x in ultime if x["xg_against"] is not None]
                        riga[f"forma{w}_xg_for_mean"] = (sum(xg_vals_for) / len(xg_vals_for)) if xg_vals_for else None
                        riga[f"forma{w}_xg_against_mean"] = (sum(xg_vals_against) / len(xg_vals_against)) if xg_vals_against else None
                righe_out.append(riga)

            # 2. Aggiorna lo storico con i risultati di QUESTA giornata
            partite_giornata = [p for p in partite if p["giornata"] == giornata]
            for p in partite_giornata:
                n_match_totale += 1
                casa, ospite = p["squadra_casa"], p["squadra_ospite"]
                gol_casa, gol_ospite = p["gol_casa"], p["gol_ospite"]

                team_casa_u = norm_team_understat(casa)
                team_ospite_u = norm_team_understat(ospite)
                xg_pair = trova_xg(xg_sum, stagione, p["data"], team_casa_u, team_ospite_u)
                if xg_pair:
                    xg_casa, xg_ospite = xg_pair
                    n_match_xg += 1
                else:
                    xg_casa, xg_ospite = None, None

                if gol_casa > gol_ospite:
                    punti_casa, punti_ospite = 3, 0
                elif gol_casa < gol_ospite:
                    punti_casa, punti_ospite = 0, 3
                else:
                    punti_casa, punti_ospite = 1, 1

                storico[casa].append({
                    "punti": punti_casa, "gf": gol_casa, "gs": gol_ospite,
                    "xg_for": xg_casa, "xg_against": xg_ospite,
                })
                storico[ospite].append({
                    "punti": punti_ospite, "gf": gol_ospite, "gs": gol_casa,
                    "xg_for": xg_ospite, "xg_against": xg_casa,
                })

        log.info("%s: %d giornate, %d squadre, forma dinamica calcolata", stagione, len(giornate_presenti), len(squadre))

    log.info("Partite totali: %d, con xG agganciato: %d (%.1f%%)",
              n_match_totale, n_match_xg, 100 * n_match_xg / n_match_totale if n_match_totale else 0)
    return righe_out


def main():
    partite_per_stagione = carica_partite_univoche()
    log.info("Stagioni caricate: %s", sorted(partite_per_stagione.keys()))
    for stagione, partite in partite_per_stagione.items():
        log.info("%s: %d partite univoche", stagione, len(partite))

    xg_sum = carica_xg_partite()

    righe_out = calcola_forma_dinamica(partite_per_stagione, xg_sum)

    fieldnames = ["stagione", "giornata", "squadra"]
    for w in FINESTRE:
        fieldnames += [
            f"forma{w}_punti_mean", f"forma{w}_gf_mean", f"forma{w}_gs_mean",
            f"forma{w}_xg_for_mean", f"forma{w}_xg_against_mean", f"forma{w}_n_partite",
        ]

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in righe_out:
            writer.writerow(r)

    log.info("Scritte %d righe in %s", len(righe_out), OUT_PATH)


if __name__ == "__main__":
    main()
