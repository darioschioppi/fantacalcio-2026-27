#!/usr/bin/env python3
"""
Applica i 4 modelli di rendimento stagionale (fantamedia/gol/assist/bonus)
alla rosa 2026-27 del Como, come richiesto da Dario ("Testiamo il modello
per tutti i giocatori del Como la fantamedia previste per 2026-27").

Riusa la stessa logica di feature engineering di
build_stagione_giocatore_dataset.py (stessa definizione di "presenza
valida", stesse metriche lag1/ma3/career_mean, stesso principio
anti-leakage: solo dati fino a fine 2025-26 + quotazione iniziale 2026-27
che è pubblicata prima del campionato quindi non leakage), ma costruisce
le righe SOLO per i giocatori attualmente in rosa Como secondo le
quotazioni 2026-27, usando stagione_n1=2025-26 (l'unica stagione
precedente disponibile per un giocatore mai visto prima con presenze
Serie A: nessuna riga generata, dichiarato esplicitamente "non
prevedibile" invece di inventare un valore).

AGGIORNAMENTO (richiesta Dario dopo il primo test: "valutasse anche lo
storico del giocatore, età e il fatto che il Como vada in Champions"):
aggiunte le stesse feature di arricchimento v2 del builder -
`{metrica}_career_mean` (media su TUTTE le stagioni precedenti
disponibili, non solo le ultime 3), `eta_n1` (da
eta_giocatori_storico_2015_2026.csv, Transfermarkt), `squadra_in_champions_n1`
(Como NON era in Champions 2025-26) e `squadra_in_champions_target` (Como
sarà in Champions 2026-27, qualificazione nota prima che la stagione
cominci - non leakage, stesso principio già usato per
quotazione_iniziale_target).

AGGIORNAMENTO 2 (v3, richiesta Dario dal PDF 327 variabili + scraping
infortuni/profilo da Transfermarkt): aggiunte infortuni_n1_count,
infortuni_n1_giorni_totali, infortuni_career_count, altezza_m,
nazionalita, piede_dominante - stessa logica del builder v3
(infortuni_giocatori_storico_2015_2026.csv/profilo_giocatori_storico_2015_2026.csv).

Output: stampa a schermo una tabella giocatore/ruolo/quotazione/previsioni,
più log su work/data/predict_como_2026_27_log.txt
"""
import csv
import logging
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
UNDERSTAT_PATH = DATA_DIR / "understat_player_match_stats_storico_2015_2026.csv"
CLASSIFICA_PATH = DATA_DIR / "classifica_dinamica_storico_2015_2026.csv"
QUOTAZIONI_PATH = DATA_DIR / "quotazioni_fantacalcio_storico_2015_2026.csv"
ETA_PATH = DATA_DIR / "eta_giocatori_storico_2015_2026.csv"
INFORTUNI_PATH = DATA_DIR / "infortuni_giocatori_storico_2015_2026.csv"
PROFILO_PATH = DATA_DIR / "profilo_giocatori_storico_2015_2026.csv"
LOG_PATH = DATA_DIR / "predict_como_2026_27_log.txt"

# Stesse squadre italiane in Champions League di build_stagione_giocatore_dataset.py
# (Como e' qualificato per la 2026-27, verificato via ricerca web da Dario/contesto lega)
SQUADRE_CHAMPIONS_TARGET = {"Inter", "Napoli", "Roma", "Como"}
SQUADRA_CHAMPIONS_N1 = set()  # Como NON era in Champions nel 2025-26

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("predict_como_2026_27")

STAGIONI_STORICHE = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                      "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
STAGIONE_IDX = {s: i for i, s in enumerate(STAGIONI_STORICHE)}
STAGIONE_TARGET = "2026-27"
STAGIONE_N1 = "2025-26"

METRICHE_BASE = ["fantamedia", "gol", "assist", "bonus_netti", "presenze",
                  "voto_medio", "minuti_totali", "xg_totale", "xa_totale", "shots_totali"]

TARGETS = ["fantamedia", "gol", "assist", "bonus_netti"]

# stesso ordine di feature usato in training (train_model_rendimento_stagionale.py,
# copiato dal log "Feature usate" per garantire corrispondenza esatta col
# modello v2 rialllenato con career_mean/eta_n1/squadra_in_champions):
FEATURE_ORDER = (
    ["ruolo", "cambio_squadra", "presenze_cumulate_carriera_n1"]
    + [f"{m}_lag1" for m in METRICHE_BASE]
    + [f"{m}_ma3" for m in METRICHE_BASE]
    + [f"{m}_career_mean" for m in METRICHE_BASE]
    + ["eta_n1", "squadra_in_champions_n1", "squadra_in_champions_target",
       "squadra_punti_finali_n1", "squadra_posizione_finale_n1",
       "squadra_nuova_punti_finali_n1", "squadra_nuova_posizione_finale_n1",
       "quotazione_iniziale_n1", "quotazione_attuale_n1", "fvm_n1",
       "quotazione_iniziale_target",
       "infortuni_n1_count", "infortuni_n1_giorni_totali",
       "infortuni_career_count", "altezza_m", "nazionalita", "piede_dominante"]
)


def to_float(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def carica_presenze_valide():
    presenze = []
    with open(VOTI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["fonte_voto"] != "redazione":
                continue
            if str(row["senza_voto"]) == "True":
                continue
            voto = to_float(row["voto"])
            fantavoto = to_float(row["fantavoto"])
            if voto is None or fantavoto is None:
                continue
            stagione = row["stagione"]
            if stagione not in STAGIONE_IDX:
                continue
            presenze.append({
                "stagione": stagione,
                "player_id": row["player_id"],
                "nome_giocatore": row["nome_giocatore"],
                "squadra_giocatore": row["squadra_giocatore"],
                "ruolo": row["ruolo"],
                "voto": voto,
                "fantavoto": fantavoto,
                "gol_fatti": to_float(row["gol_fatti"]) or 0.0,
                "assist": to_float(row["assist"]) or 0.0,
            })
    log.info("Presenze valide caricate: %d", len(presenze))
    return presenze


def carica_understat_per_stagione():
    agg = defaultdict(lambda: {"minuti_totali": 0.0, "xg_totale": 0.0, "xa_totale": 0.0, "shots_totali": 0.0})
    with open(UNDERSTAT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stagione = row["stagione"]
            player_id = row["player_id"]
            if stagione not in STAGIONE_IDX or not player_id:
                continue
            key = (player_id, stagione)
            agg[key]["minuti_totali"] += to_float(row.get("time")) or 0.0
            agg[key]["xg_totale"] += to_float(row.get("xG")) or 0.0
            agg[key]["xa_totale"] += to_float(row.get("xA")) or 0.0
            agg[key]["shots_totali"] += to_float(row.get("shots")) or 0.0
    return agg


def carica_classifica_finale():
    per_key_giornata_max = {}
    with open(CLASSIFICA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stagione = row["stagione"]
            squadra = row["squadra"]
            giornata = int(row["giornata"])
            key = (stagione, squadra)
            if key not in per_key_giornata_max or giornata > per_key_giornata_max[key][0]:
                per_key_giornata_max[key] = (giornata, {
                    "punti_pre": to_float(row["punti_pre"]),
                    "posizione_pre": to_float(row["posizione_pre"]),
                })
    return {key: v[1] for key, v in per_key_giornata_max.items()}


def carica_quotazioni():
    quot = {}
    with open(QUOTAZIONI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            quot[(row["player_id"], row["stagione"])] = {
                "quotazione_iniziale": to_float(row["quotazione_iniziale"]),
                "quotazione_attuale": to_float(row["quotazione_attuale"]),
                "fvm": to_float(row["fvm"]),
                "nome_giocatore": row["nome_giocatore"],
                "ruolo_classic": row["ruolo_classic"],
                "squadra_sigla": row["squadra_sigla"],
            }
    return quot


def aggrega_per_giocatore_stagione(presenze, understat_agg):
    per_key = defaultdict(list)
    for p in presenze:
        per_key[(p["player_id"], p["stagione"])].append(p)

    aggregati = {}
    for (player_id, stagione), pres in per_key.items():
        n = len(pres)
        fantamedia = sum(p["fantavoto"] for p in pres) / n
        voto_medio = sum(p["voto"] for p in pres) / n
        gol = sum(p["gol_fatti"] for p in pres)
        assist = sum(p["assist"] for p in pres)
        bonus_netti = sum(p["fantavoto"] - p["voto"] for p in pres)
        u = understat_agg.get((player_id, stagione), {})
        aggregati[(player_id, stagione)] = {
            "fantamedia": fantamedia, "voto_medio": voto_medio, "gol": gol,
            "assist": assist, "bonus_netti": bonus_netti, "presenze": float(n),
            "minuti_totali": u.get("minuti_totali", 0.0),
            "xg_totale": u.get("xg_totale", 0.0),
            "xa_totale": u.get("xa_totale", 0.0),
            "shots_totali": u.get("shots_totali", 0.0),
            "ruolo": max(set(p["ruolo"] for p in pres), key=lambda r: sum(1 for p in pres if p["ruolo"] == r)),
            "squadra_giocatore": pres[-1]["squadra_giocatore"],
            "nome_giocatore": pres[-1]["nome_giocatore"],
        }
    return aggregati


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def carica_eta():
    """dict {(player_id, stagione): eta_al_1_agosto} - stessa fonte del
    builder (scrape_eta_giocatori.py/Transfermarkt)."""
    eta = {}
    if not ETA_PATH.exists():
        log.warning("File eta' giocatori non trovato: eta_n1 sara' sempre None")
        return eta
    with open(ETA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = to_float(row["eta_al_1_agosto"])
            if v is not None:
                eta[(row["player_id"], row["stagione"])] = v
    return eta


def carica_infortuni():
    """dict {player_id: [{'stagione':.., 'giorni_stop':..}, ...]} - stessa
    fonte/logica del builder v3 (scrape_infortuni_profilo_giocatori.py)."""
    per_player = defaultdict(list)
    if not INFORTUNI_PATH.exists():
        log.warning("File infortuni non trovato: infortuni_* saranno sempre 0/None")
        return per_player
    with open(INFORTUNI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stagione = row["stagione"]
            if not stagione or stagione not in STAGIONE_IDX:
                continue
            per_player[row["player_id"]].append({
                "stagione": stagione,
                "giorni_stop": to_float(row["giorni_stop"]) or 0.0,
            })
    return per_player


def carica_profilo():
    """dict {player_id: {'altezza_m':.., 'nazionalita':.., 'piede_dominante':..}}
    - stessa fonte/logica del builder v3."""
    profilo = {}
    if not PROFILO_PATH.exists():
        log.warning("File profilo non trovato: altezza_m/nazionalita/piede_dominante saranno sempre None")
        return profilo
    with open(PROFILO_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profilo[row["player_id"]] = {
                "altezza_m": to_float(row["altezza_m"]),
                "nazionalita": row["nazionalita"] or None,
                "piede_dominante": row["piede_dominante"] or None,
            }
    return profilo


def main():
    quotazioni = carica_quotazioni()
    presenze = carica_presenze_valide()
    understat_agg = carica_understat_per_stagione()
    classifica_finale = carica_classifica_finale()
    eta_map = carica_eta()
    infortuni_map = carica_infortuni()
    profilo_map = carica_profilo()
    aggregati = aggrega_per_giocatore_stagione(presenze, understat_agg)

    per_player = defaultdict(dict)
    for (player_id, stagione), agg in aggregati.items():
        per_player[player_id][stagione] = agg

    como_players = [
        (pid, q) for (pid, s), q in quotazioni.items()
        if s == STAGIONE_TARGET and q["squadra_sigla"] == "COM"
    ]
    log.info("Giocatori Como 2026-27 (da quotazioni): %d", len(como_players))

    models = {}
    for t in TARGETS:
        path = MODELS_DIR / f"lgbm_{t}_stagionale_v1.txt"
        models[t] = lgb.Booster(model_file=str(path))

    risultati = []
    non_prevedibili = []

    for player_id, q_target in como_players:
        agg_n1 = per_player.get(player_id, {}).get(STAGIONE_N1)
        if agg_n1 is None:
            non_prevedibili.append(q_target["nome_giocatore"])
            continue

        stagioni_giocatore = sorted(per_player[player_id].keys(), key=lambda s: STAGIONE_IDX[s])
        idx_n1 = STAGIONE_IDX[STAGIONE_N1]

        stagioni_prec = []
        for back in (0, 1, 2):  # N-1, N-2, N-3 rispetto al target (N-1 stesso incluso qui come back=0 su idx_n1)
            idx_prec = idx_n1 - back
            if idx_prec < 0:
                break
            s_prec = STAGIONI_STORICHE[idx_prec]
            if s_prec in per_player[player_id]:
                stagioni_prec.append(per_player[player_id][s_prec])

        cambio_squadra = 1.0 if agg_n1["squadra_giocatore"] != "Como" else 0.0
        riga = {
            "ruolo": agg_n1["ruolo"],
            "cambio_squadra": cambio_squadra,
            "presenze_cumulate_carriera_n1": float(sum(1 for s in stagioni_giocatore if STAGIONE_IDX[s] <= idx_n1)),
        }
        for m in METRICHE_BASE:
            riga[f"{m}_lag1"] = agg_n1[m]
        for m in METRICHE_BASE:
            riga[f"{m}_ma3"] = mean_or_none([s[m] for s in stagioni_prec])

        # career_mean: media su TUTTE le stagioni precedenti disponibili
        # (non solo le ultime 3 come ma3) - stessa logica del builder v2.
        stagioni_tutte_precedenti = [
            per_player[player_id][s] for s in stagioni_giocatore if STAGIONE_IDX[s] <= idx_n1
        ]
        for m in METRICHE_BASE:
            riga[f"{m}_career_mean"] = mean_or_none([s[m] for s in stagioni_tutte_precedenti])

        # eta' al 1 agosto 2025 (stagione N-1 = 2025-26), da Transfermarkt.
        # None se il giocatore non e' stato matchato - dichiarato, non
        # inventato.
        riga["eta_n1"] = eta_map.get((player_id, STAGIONE_N1))

        # flag Champions League: Como NON era in Champions nel 2025-26,
        # ma SARA' in Champions nel 2026-27 (qualificazione nota prima che
        # la stagione cominci - non leakage, stesso principio di
        # quotazione_iniziale_target).
        riga["squadra_in_champions_n1"] = 1.0 if agg_n1["squadra_giocatore"] in SQUADRA_CHAMPIONS_N1 else 0.0
        riga["squadra_in_champions_target"] = 1.0 if "Como" in SQUADRE_CHAMPIONS_TARGET else 0.0

        # infortuni/profilo (v3): stessa logica del builder - episodi N-1 e
        # cumulati di carriera (< N), profilo "attuale" (non storico per
        # stagione, semplificazione dichiarata anche nel builder).
        episodi_giocatore = infortuni_map.get(player_id, [])
        episodi_n1 = [e for e in episodi_giocatore if e["stagione"] == STAGIONE_N1]
        riga["infortuni_n1_count"] = float(len(episodi_n1))
        riga["infortuni_n1_giorni_totali"] = sum(e["giorni_stop"] for e in episodi_n1)
        episodi_career = [e for e in episodi_giocatore if STAGIONE_IDX.get(e["stagione"], 999) <= idx_n1]
        riga["infortuni_career_count"] = float(len(episodi_career))
        prof = profilo_map.get(player_id)
        riga["altezza_m"] = prof["altezza_m"] if prof else None
        riga["nazionalita"] = prof["nazionalita"] if prof else None
        riga["piede_dominante"] = prof["piede_dominante"] if prof else None

        ctx_n1 = classifica_finale.get((STAGIONE_N1, agg_n1["squadra_giocatore"]))
        riga["squadra_punti_finali_n1"] = ctx_n1["punti_pre"] if ctx_n1 else None
        riga["squadra_posizione_finale_n1"] = ctx_n1["posizione_pre"] if ctx_n1 else None
        if cambio_squadra == 1.0:
            ctx_new = classifica_finale.get((STAGIONE_N1, "Como"))
            riga["squadra_nuova_punti_finali_n1"] = ctx_new["punti_pre"] if ctx_new else None
            riga["squadra_nuova_posizione_finale_n1"] = ctx_new["posizione_pre"] if ctx_new else None
        else:
            riga["squadra_nuova_punti_finali_n1"] = None
            riga["squadra_nuova_posizione_finale_n1"] = None

        quot_n1 = quotazioni.get((player_id, STAGIONE_N1))
        riga["quotazione_iniziale_n1"] = quot_n1["quotazione_iniziale"] if quot_n1 else None
        riga["quotazione_attuale_n1"] = quot_n1["quotazione_attuale"] if quot_n1 else None
        fvm_n1_raw = quot_n1["fvm"] if quot_n1 else None
        riga["fvm_n1"] = (fvm_n1_raw / 2.0) if fvm_n1_raw is not None else None
        riga["quotazione_iniziale_target"] = q_target["quotazione_iniziale"]

        import pandas as pd
        import numpy as np
        CAT_COLS = ["ruolo", "nazionalita", "piede_dominante"]
        X = pd.DataFrame([riga])[FEATURE_ORDER]
        for c in X.columns:
            if c not in CAT_COLS:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in CAT_COLS:
            X[c] = X[c].astype("category")

        pred = {}
        for t in TARGETS:
            pred[t] = float(models[t].predict(X)[0])

        risultati.append({
            "nome": agg_n1["nome_giocatore"],
            "ruolo_classic": q_target["ruolo_classic"],
            "quotazione_iniziale_2026_27": q_target["quotazione_iniziale"],
            "fantamedia_lag1_2025_26": round(agg_n1["fantamedia"], 2),
            "presenze_2025_26": int(agg_n1["presenze"]),
            "pred_fantamedia": round(pred["fantamedia"], 2),
            "pred_gol": round(pred["gol"], 1),
            "pred_assist": round(pred["assist"], 1),
            "pred_bonus_netti": round(pred["bonus_netti"], 1),
        })

    risultati.sort(key=lambda r: r["quotazione_iniziale_2026_27"] or 0, reverse=True)

    log.info("=== Previsioni Como 2026-27 (%d giocatori con storico 2025-26) ===", len(risultati))
    header = f"{'Nome':<22}{'Ruolo':<6}{'QI26-27':>8}{'FM25-26':>9}{'Pres25-26':>10}{'PredFM':>8}{'PredGol':>8}{'PredAst':>8}{'PredBon':>8}"
    log.info(header)
    for r in risultati:
        log.info(f"{r['nome']:<22}{r['ruolo_classic']:<6}{r['quotazione_iniziale_2026_27']:>8.0f}"
                  f"{r['fantamedia_lag1_2025_26']:>9.2f}{r['presenze_2025_26']:>10d}"
                  f"{r['pred_fantamedia']:>8.2f}{r['pred_gol']:>8.1f}{r['pred_assist']:>8.1f}{r['pred_bonus_netti']:>8.1f}")

    log.info("")
    log.info("Giocatori Como 2026-27 SENZA storico Serie A 2025-26 (non prevedibili, %d): %s",
              len(non_prevedibili), non_prevedibili)


if __name__ == "__main__":
    main()
