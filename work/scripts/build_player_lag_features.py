#!/usr/bin/env python3
"""
Calcola, per ogni riga voto (una presenza = una partita valutata da una
fonte), le feature "storia recente del giocatore" basate SOLO su presenze
PRECEDENTI a quella riga - mai sulla partita corrente che si vuole
predire. Questo introduce la componente autoregressiva/di forma
individuale nel modello previsionale, sostituendo le colonne
understat_* (che oggi descrivono la partita CORRENTE, cioè l'esito da
predire, non un'informazione disponibile prima del match).

Definizione di "presenza": stessa usata come target del modello,
fonte_voto == "redazione" e senza_voto != "True" (cfr. train_model.py).
Nota: si usa fonte_voto=="redazione" anche per costruire la SERIE storica
(coerenza interna - i lag di voto devono venire dalla stessa fonte che si
sta cercando di predire, altrimenti si mescolerebbero scale di giudizio
potenzialmente diverse).

Ordinamento cronologico: per ogni player_id, le presenze sono ordinate per
(stagione, giornata) - NON si resetta la storia a inizio stagione: la
forma di un giocatore a inizio stagione N+1 dipende anche da come stava
giocando a fine stagione N (assunzione ragionevole, coerente con come
funziona la forma calcistica reale; le interruzioni per cambio squadra
non vengono trattate in modo speciale in questa prima versione).

Le feature calcolate su ogni riga sono medie delle ultime K presenze
STRETTAMENTE PRECEDENTI (mai la riga corrente):
  - voto_lag_mean_3, voto_lag_mean_5      (media voto - autoregressivo)
  - understat_xG_lag_mean_5
  - understat_xA_lag_mean_5
  - understat_shots_lag_mean_5
  - understat_time_lag_mean_5
  - presenze_cumulate                      (numero di presenze STRETTAMENTE
                                             precedenti già accumulate, non
                                             solo nella finestra - proxy di
                                             esperienza/rodaggio)

Le statistiche Understat (xG/xA/shots/time) per la presenza in questione
vengono da feature_dataset_v1.csv (già unito con Understat via
build_feature_dataset.py, colonne understat_xG/xA/shots/time) - qui
vengono READ-ONLY per costruire lo storico, la partita CORRENTE non viene
mai scritta come output di questa riga (si scrive solo la media delle
K precedenti).

Le prime presenze di un giocatore, quando non ci sono ancora K presenze
precedenti disponibili, hanno lag calcolato sulle presenze disponibili (se
almeno 1) o vuoto/NaN (se zero presenze precedenti) - mai riempimento
artificioso con zero, che sarebbe un valore finto (media voto 0 non
significa "nessuna informazione", significherebbe "gioca malissimo").

Output:
  work/data/player_lag_features_storico_2015_2026.csv
  colonne: match_id, stagione, player_id, giornata,
           voto_lag_mean_3, voto_lag_mean_5,
           understat_xG_lag_mean_5, understat_xA_lag_mean_5,
           understat_shots_lag_mean_5, understat_time_lag_mean_5,
           presenze_cumulate

Uso:
  python3 build_player_lag_features.py
"""
import csv
import logging
from collections import defaultdict, deque
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURE_V1_PATH = DATA_DIR / "feature_dataset_v1.csv"
OUT_PATH = DATA_DIR / "player_lag_features_storico_2015_2026.csv"
LOG_PATH = DATA_DIR / "player_lag_features_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("player_lag_features")

MAXLEN = 5
STAGIONI_ORDINE = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                   "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
STAGIONE_IDX = {s: i for i, s in enumerate(STAGIONI_ORDINE)}


def to_float(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def carica_presenze():
    """Lista di dict con le presenze valide (fonte_voto=redazione, senza_voto!=True),
    con le colonne necessarie per l'ordinamento cronologico e le stats Understat
    della PARTITA (che diventeranno storico per le presenze FUTURE dello stesso giocatore)."""
    presenze = []
    with open(FEATURE_V1_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["fonte_voto"] != "redazione":
                continue
            if str(row["senza_voto"]) == "True":
                continue
            voto = to_float(row["voto"])
            if voto is None:
                continue
            try:
                giornata = int(row["giornata"])
            except (ValueError, TypeError):
                continue
            stagione = row["stagione"]
            if stagione not in STAGIONE_IDX:
                continue
            presenze.append({
                "match_id": row["match_id"],
                "stagione": stagione,
                "player_id": row["player_id"],
                "giornata": giornata,
                "voto": voto,
                "understat_xG": to_float(row.get("understat_xG")),
                "understat_xA": to_float(row.get("understat_xA")),
                "understat_shots": to_float(row.get("understat_shots")),
                "understat_time": to_float(row.get("understat_time")),
            })
    log.info("Presenze valide caricate: %d", len(presenze))
    return presenze


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def calcola_lag_features(presenze):
    per_player = defaultdict(list)
    for p in presenze:
        if p["player_id"]:
            per_player[p["player_id"]].append(p)

    righe_out = []
    n_con_almeno_1_lag = 0
    n_con_5_lag = 0

    for player_id, pres_giocatore in per_player.items():
        pres_giocatore.sort(key=lambda p: (STAGIONE_IDX[p["stagione"]], p["giornata"]))

        storico = deque(maxlen=MAXLEN)  # ultime presenze PRECEDENTI

        for p in pres_giocatore:
            n_prec = len(storico)
            voti_prec = [x["voto"] for x in storico]
            riga = {
                "match_id": p["match_id"],
                "stagione": p["stagione"],
                "player_id": player_id,
                "giornata": p["giornata"],
                "voto_lag_mean_3": mean_or_none(voti_prec[-3:]) if n_prec > 0 else None,
                "voto_lag_mean_5": mean_or_none(voti_prec) if n_prec > 0 else None,
                "understat_xG_lag_mean_5": mean_or_none([x["understat_xG"] for x in storico]) if n_prec > 0 else None,
                "understat_xA_lag_mean_5": mean_or_none([x["understat_xA"] for x in storico]) if n_prec > 0 else None,
                "understat_shots_lag_mean_5": mean_or_none([x["understat_shots"] for x in storico]) if n_prec > 0 else None,
                "understat_time_lag_mean_5": mean_or_none([x["understat_time"] for x in storico]) if n_prec > 0 else None,
                "presenze_cumulate": n_prec,
            }
            righe_out.append(riga)
            if n_prec >= 1:
                n_con_almeno_1_lag += 1
            if n_prec >= 5:
                n_con_5_lag += 1

            storico.append(p)

    log.info("Giocatori distinti: %d", len(per_player))
    log.info("Righe totali: %d", len(righe_out))
    log.info("Righe con almeno 1 presenza precedente: %d (%.1f%%)",
              n_con_almeno_1_lag, 100 * n_con_almeno_1_lag / len(righe_out) if righe_out else 0)
    log.info("Righe con 5 presenze precedenti (finestra piena): %d (%.1f%%)",
              n_con_5_lag, 100 * n_con_5_lag / len(righe_out) if righe_out else 0)
    return righe_out


def main():
    presenze = carica_presenze()
    righe_out = calcola_lag_features(presenze)

    fieldnames = [
        "match_id", "stagione", "player_id", "giornata",
        "voto_lag_mean_3", "voto_lag_mean_5",
        "understat_xG_lag_mean_5", "understat_xA_lag_mean_5",
        "understat_shots_lag_mean_5", "understat_time_lag_mean_5",
        "presenze_cumulate",
    ]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in righe_out:
            writer.writerow(r)

    log.info("Scritte %d righe in %s", len(righe_out), OUT_PATH)


if __name__ == "__main__":
    main()
