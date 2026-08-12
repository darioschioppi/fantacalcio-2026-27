#!/usr/bin/env python3
"""
Applica i modelli di rendimento stagionale a TUTTI i giocatori di TUTTE le
20 squadre di Serie A 2026-27 (richiesta Dario: "genera la lista di
giocatori completa della serie a 2026-27 con le previsioni fantamedia voto
e gol e assist e partite titolari").

Generalizzazione di predict_como_2026_27.py (stessa logica di feature
engineering, stesso principio anti-leakage: solo dati fino a fine 2025-26 +
quotazione iniziale 2026-27 che e' pubblicata prima del campionato quindi
non leakage) rimuovendo il filtro sulla sola squadra Como.

TARGET (7 modelli LightGBM separati, v4):
  - fantamedia          (media fantavoto redazione, con bonus/malus)
  - voto                (media voto redazione puro, SENZA bonus/malus -
                          nuovo in questa fase: "voto" non era un target
                          allenato finora; qui e' un target dichiarato a
                          livello STAGIONALE aggregato, diverso dal voto di
                          singola partita gia' dimostrato non predicibile
                          nella fase precedente)
  - gol
  - assist
  - bonus_netti
  - presenze            (presenze totali previste, contesto)
  - presenze_titolare   ("partite titolari": righe Understat con
                          position != 'Sub' - vedi nota sotto)

NOTA su "partite titolari": inizialmente NON era calcolabile in modo
affidabile perche' build_stagione_giocatore_dataset.py univa gli aggregati
Understat (xG/xA/tiri/minuti/titolare-o-no) usando direttamente il
player_id di fantacalcio.it come se fosse lo stesso ID di Understat -
NON lo e' (due spazi ID indipendenti, bug di join pre-esistente scoperto
in questa fase: solo ~2.5% di match reali). CORRETTO in questa fase
usando il bridge gia' esistente `player_name_mapping.csv`
(stagione,squadra,nome_fantacalcio -> player_id_understat, la stessa
mappa gia' usata correttamente in build_feature_dataset.py) - copertura
salita al 98.7%. Con il join corretto, la colonna Understat 'position'
(valore 'Sub' per i subentrati, verificato: mediana 17 min per 'Sub' vs 90
min per le altre posizioni) diventa utilizzabile: presenze_titolare conta
le righe con position != 'Sub' per (player_id, stagione).

Giocatori 2026-27 SENZA presenze valide Serie A nella stagione 2025-26
(esordienti, rientri da altre leghe/prestiti all'estero, nuovi acquisti
dall'estero) NON vengono previsti - dichiarati esplicitamente in una
sezione separata invece di inventare una previsione senza base storica.

Output:
  - work/data/previsioni_serie_a_2026_27.csv (tutti i giocatori previsti,
    ordinati per squadra e quotazione)
  - log su work/data/predict_serie_a_2026_27_log.txt (incluso elenco
    giocatori non prevedibili per squadra)
"""
import csv
import logging
import os
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# DATA_DIR/MODELS_DIR sono sovrascrivibili via env var (usato dalla test suite in
# work/tests per puntare a fixture/modelli di test); default invariato se non impostate.
DATA_DIR = Path(os.environ.get("FANTACALCIO_DATA_DIR") or (Path(__file__).resolve().parent.parent / "data"))
MODELS_DIR = Path(os.environ.get("FANTACALCIO_MODELS_DIR") or (Path(__file__).resolve().parent.parent / "models"))
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
UNDERSTAT_PATH = DATA_DIR / "understat_player_match_stats_storico_2015_2026.csv"
CLASSIFICA_PATH = DATA_DIR / "classifica_dinamica_storico_2015_2026.csv"
QUOTAZIONI_PATH = DATA_DIR / "quotazioni_fantacalcio_storico_2015_2026.csv"
ETA_PATH = DATA_DIR / "eta_giocatori_storico_2015_2026.csv"
INFORTUNI_PATH = DATA_DIR / "infortuni_giocatori_storico_2015_2026.csv"
PROFILO_PATH = DATA_DIR / "profilo_giocatori_storico_2015_2026.csv"
FORUM_ESPERTI_PATH = DATA_DIR / "forum_esperti_pagelle_2026_27.csv"
MAPPING_PATH = DATA_DIR / "player_name_mapping.csv"
OUT_PATH = DATA_DIR / "previsioni_serie_a_2026_27.csv"
LOG_PATH = DATA_DIR / "predict_serie_a_2026_27_log.txt"

# fix bug join Understat (v4): stessa mappa nomi squadra fantacalcio->Understat
# di build_stagione_giocatore_dataset.py/build_feature_dataset.py.
TEAM_NAME_MAP_UNDERSTAT = {
    "SPAL": "SPAL 2013",
    "Milan": "AC Milan",
    "Parma": "Parma Calcio 1913",
}

# Stesse squadre italiane in Champions League 2026-27 di
# build_stagione_giocatore_dataset.py (verificato via ricerca web).
SQUADRE_CHAMPIONS_TARGET = {"Inter", "Napoli", "Roma", "Como"}
SQUADRE_CHAMPIONS_N1 = {"Napoli", "Inter", "Atalanta", "Juventus"}  # 2025-26

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("predict_serie_a_2026_27")

STAGIONI_STORICHE = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                      "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
STAGIONE_IDX = {s: i for i, s in enumerate(STAGIONI_STORICHE)}
STAGIONE_TARGET = "2026-27"
STAGIONE_N1 = "2025-26"

METRICHE_BASE = ["fantamedia", "gol", "assist", "bonus_netti", "presenze",
                  "voto_medio", "minuti_totali", "xg_totale", "xa_totale",
                  "shots_totali", "presenze_titolare",
                  "ammonizioni_totali", "espulsioni_totali",
                  "rigori_segnati_totali", "rigori_sbagliati_totali",
                  "rigori_parati_totali"]

# nome file modello -> nome colonna output (voto_medio ha nome file
# "voto_medio", non "voto", perche' il builder/training usano
# voto_medio_target per essere inequivocabili rispetto a "voto" generico)
TARGETS = {
    "fantamedia": "fantamedia",
    "voto_medio": "voto",
    "gol": "gol",
    "assist": "assist",
    "bonus_netti": "bonus_netti",
    "presenze": "presenze_previste",
    "presenze_titolare": "presenze_titolare_previste",
}

# Sigla squadra quotazioni -> nome squadra usato in voti_storici/classifica
# (stessa mappa di build_stagione_giocatore_dataset.py)
SIGLA_TO_NOME = {
    "ATA": "Atalanta", "BOL": "Bologna", "CAR": "Carpi", "CHI": "Chievo",
    "EMP": "Empoli", "FIO": "Fiorentina", "FRO": "Frosinone", "GEN": "Genoa",
    "INT": "Inter", "JUV": "Juventus", "LAZ": "Lazio", "MIL": "Milan",
    "NAP": "Napoli", "PAL": "Palermo", "ROM": "Roma", "SAM": "Sampdoria",
    "SAS": "Sassuolo", "TOR": "Torino", "UDI": "Udinese", "VER": "Verona",
    "CRO": "Crotone", "BEN": "Benevento", "SPA": "SPAL", "PES": "Pescara",
    "PAR": "Parma", "BRE": "Brescia", "LEC": "Lecce", "CAG": "Cagliari",
    "SPE": "Spezia", "SAL": "Salernitana", "VEN": "Venezia", "COM": "Como",
    "MON": "Monza", "CRE": "Cremonese", "PIS": "Pisa",
}

# ATTENZIONE ORDINE: deve corrispondere ESATTAMENTE all'ordine colonne del
# dataset costruito da build_stagione_giocatore_dataset.py (verificato dal
# log "Feature usate" di train_model_rendimento_stagionale.py) - LightGBM
# Booster.predict() su un DataFrame usa l'ordine POSIZIONALE delle colonne,
# non un match per nome. std3/trend3 sono INTERLEAVED per metrica
# (fantamedia_std3, fantamedia_trend3, gol_std3, gol_trend3, ...), NON
# "tutti gli std3 poi tutti i trend3" - stesso ordine del doppio loop in
# costruisci_righe() del builder.
_STD_TREND_INTERLEAVED = [x for m in METRICHE_BASE for x in (f"{m}_std3", f"{m}_trend3")]
FEATURE_ORDER = (
    ["ruolo", "cambio_squadra", "presenze_cumulate_carriera_n1"]
    + [f"{m}_lag1" for m in METRICHE_BASE]
    + [f"{m}_ma3" for m in METRICHE_BASE]
    + _STD_TREND_INTERLEAVED
    + [f"{m}_career_mean" for m in METRICHE_BASE]
    + ["eta_n1", "squadra_in_champions_n1", "squadra_in_champions_target",
       "squadra_punti_finali_n1", "squadra_posizione_finale_n1",
       "squadra_nuova_punti_finali_n1", "squadra_nuova_posizione_finale_n1",
       "quotazione_iniziale_n1", "quotazione_attuale_n1", "fvm_n1",
       "quotazione_iniziale_target",
       "infortuni_n1_count", "infortuni_n1_giorni_totali",
       "infortuni_career_count", "altezza_m", "nazionalita", "piede_dominante",
       "titolarita_forum", "media_voto_forum", "salute_forum", "bonus_forum",
       "consiglio_esperti_forum", "totale_forum"]
)
CAT_COLS = ["ruolo", "nazionalita", "piede_dominante"]


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
                # v6: cartellini/rigori (stessa logica del builder)
                "ammonizione": 1.0 if str(row["ammonizione"]) == "True" else 0.0,
                "espulsione": 1.0 if str(row["espulsione"]) == "True" else 0.0,
                "rigori_segnati": to_float(row["rigori_segnati"]) or 0.0,
                "rigori_sbagliati": to_float(row["rigori_sbagliati"]) or 0.0,
                "rigori_parati": to_float(row["rigori_parati"]) or 0.0,
            })
    log.info("Presenze valide caricate: %d", len(presenze))
    return presenze


def carica_understat_per_stagione():
    """Indicizzato per player_id_understat (ID interno Understat), NON per
    player_id fantacalcio - vedi carica_name_mapping/trova_understat_agg
    per il bridge corretto (fix bug join v4)."""
    agg = defaultdict(lambda: {"minuti_totali": 0.0, "xg_totale": 0.0, "xa_totale": 0.0,
                                 "shots_totali": 0.0, "presenze_titolare": 0.0})
    with open(UNDERSTAT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stagione = row["stagione"]
            player_id_understat = row["player_id"]
            if stagione not in STAGIONE_IDX or not player_id_understat:
                continue
            key = (player_id_understat, stagione)
            agg[key]["minuti_totali"] += to_float(row.get("time")) or 0.0
            agg[key]["xg_totale"] += to_float(row.get("xG")) or 0.0
            agg[key]["xa_totale"] += to_float(row.get("xA")) or 0.0
            agg[key]["shots_totali"] += to_float(row.get("shots")) or 0.0
            if row.get("position") != "Sub":
                agg[key]["presenze_titolare"] += 1.0
    return agg


def carica_name_mapping():
    """dict {(stagione, squadra_fantacalcio, nome_fantacalcio): player_id_understat}
    - bridge corretto tra i due spazi ID (fix bug join v4)."""
    mapping = {}
    if not MAPPING_PATH.exists():
        log.warning("File player_name_mapping.csv non trovato: xg/xa/shots/minuti/titolare saranno sempre 0.0")
        return mapping
    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[(row["stagione"], row["squadra"], row["nome_fantacalcio"])] = row["player_id_understat"]
    return mapping


def norm_team_understat(name):
    return TEAM_NAME_MAP_UNDERSTAT.get(name, name)


def trova_understat_agg(name_mapping, understat_agg, stagione, pres):
    combinazioni = {(p["nome_giocatore"], p["squadra_giocatore"]) for p in pres}
    for nome, squadra in combinazioni:
        squadra_norm = norm_team_understat(squadra)
        player_id_understat = name_mapping.get((stagione, squadra_norm, nome))
        if player_id_understat:
            u = understat_agg.get((player_id_understat, stagione))
            if u:
                return u
    return {}


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


def aggrega_per_giocatore_stagione(presenze, understat_agg, name_mapping):
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
        ammonizioni_totali = sum(p["ammonizione"] for p in pres)
        espulsioni_totali = sum(p["espulsione"] for p in pres)
        rigori_segnati_totali = sum(p["rigori_segnati"] for p in pres)
        rigori_sbagliati_totali = sum(p["rigori_sbagliati"] for p in pres)
        rigori_parati_totali = sum(p["rigori_parati"] for p in pres)
        u = trova_understat_agg(name_mapping, understat_agg, stagione, pres)
        aggregati[(player_id, stagione)] = {
            "fantamedia": fantamedia, "voto_medio": voto_medio, "gol": gol,
            "assist": assist, "bonus_netti": bonus_netti, "presenze": float(n),
            "minuti_totali": u.get("minuti_totali", 0.0),
            "xg_totale": u.get("xg_totale", 0.0),
            "xa_totale": u.get("xa_totale", 0.0),
            "shots_totali": u.get("shots_totali", 0.0),
            "presenze_titolare": u.get("presenze_titolare", 0.0),
            "ammonizioni_totali": ammonizioni_totali,
            "espulsioni_totali": espulsioni_totali,
            "rigori_segnati_totali": rigori_segnati_totali,
            "rigori_sbagliati_totali": rigori_sbagliati_totali,
            "rigori_parati_totali": rigori_parati_totali,
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


def std_or_none(values):
    """Stessa logica di build_stagione_giocatore_dataset.py (v5): deviazione
    standard di popolazione sulla stessa finestra fino-3 stagioni precedenti
    usata per _ma3. None se <2 valori."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def trend_or_none(values_cronologiche):
    """Stessa logica di build_stagione_giocatore_dataset.py (v5): pendenza
    di regressione lineare semplice, values_cronologiche in ordine dal piu'
    vecchio al piu' recente. None se <2 valori."""
    vals = [v for v in values_cronologiche if v is not None]
    n = len(vals)
    if n < 2:
        return None
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(vals) / n
    num = sum((xs[i] - x_mean) * (vals[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def carica_eta():
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


def carica_forum_esperti():
    """dict {player_id: {colonna_forum: valore}} da
    scrape_forum_esperti_2026_27.py - stessa fonte/logica di
    build_stagione_giocatore_dataset.py, qui applicata direttamente a
    tutte le righe (in questo script TUTTE le righe sono 2026-27 per
    costruzione, quindi non serve il filtro per stagione_target usato
    nel builder)."""
    forum = {}
    if not FORUM_ESPERTI_PATH.exists():
        log.warning("File forum esperti non trovato: forum_* saranno sempre None")
        return forum
    with open(FORUM_ESPERTI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = row.get("player_id")
            if not player_id or player_id in forum:
                continue
            forum[player_id] = {
                "titolarita_forum": to_float(row["titolarita_forum"]),
                "media_voto_forum": to_float(row["media_voto_forum"]),
                "salute_forum": to_float(row["salute_forum"]),
                "bonus_forum": to_float(row["bonus_forum"]),
                "consiglio_esperti_forum": to_float(row["consiglio_esperti_forum"]),
                "totale_forum": to_float(row["totale_forum"]),
            }
    log.info("Pagelle forum esperti caricate per %d player_id", len(forum))
    return forum


def main():
    quotazioni = carica_quotazioni()
    presenze = carica_presenze_valide()
    understat_agg = carica_understat_per_stagione()
    name_mapping = carica_name_mapping()
    classifica_finale = carica_classifica_finale()
    eta_map = carica_eta()
    infortuni_map = carica_infortuni()
    profilo_map = carica_profilo()
    forum_map = carica_forum_esperti()
    aggregati = aggrega_per_giocatore_stagione(presenze, understat_agg, name_mapping)

    per_player = defaultdict(dict)
    for (player_id, stagione), agg in aggregati.items():
        per_player[player_id][stagione] = agg

    tutti_2026_27 = [
        (pid, q) for (pid, s), q in quotazioni.items() if s == STAGIONE_TARGET
    ]
    squadre = sorted(set(q["squadra_sigla"] for _, q in tutti_2026_27))
    log.info("Giocatori 2026-27 totali (da quotazioni): %d, squadre: %d (%s)",
              len(tutti_2026_27), len(squadre), squadre)

    models = {}
    for model_file in TARGETS:
        path = MODELS_DIR / f"lgbm_{model_file}_stagionale_v1.txt"
        models[model_file] = lgb.Booster(model_file=str(path))

    risultati = []
    non_prevedibili_per_squadra = defaultdict(list)

    for player_id, q_target in tutti_2026_27:
        squadra_sigla = q_target["squadra_sigla"]
        agg_n1 = per_player.get(player_id, {}).get(STAGIONE_N1)
        if agg_n1 is None:
            non_prevedibili_per_squadra[squadra_sigla].append(q_target["nome_giocatore"])
            continue

        stagioni_giocatore = sorted(per_player[player_id].keys(), key=lambda s: STAGIONE_IDX[s])
        idx_n1 = STAGIONE_IDX[STAGIONE_N1]

        stagioni_prec = []
        for back in (0, 1, 2):
            idx_prec = idx_n1 - back
            if idx_prec < 0:
                break
            s_prec = STAGIONI_STORICHE[idx_prec]
            if s_prec in per_player[player_id]:
                stagioni_prec.append(per_player[player_id][s_prec])

        squadra_nome_target = SIGLA_TO_NOME.get(squadra_sigla, squadra_sigla)
        cambio_squadra = 1.0 if agg_n1["squadra_giocatore"] != squadra_nome_target else 0.0
        riga = {
            "ruolo": agg_n1["ruolo"],
            "cambio_squadra": cambio_squadra,
            "presenze_cumulate_carriera_n1": float(sum(1 for s in stagioni_giocatore if STAGIONE_IDX[s] <= idx_n1)),
        }
        for m in METRICHE_BASE:
            riga[f"{m}_lag1"] = agg_n1[m]
        for m in METRICHE_BASE:
            riga[f"{m}_ma3"] = mean_or_none([s[m] for s in stagioni_prec])

        # AGGIORNAMENTO v5 (richiesta Dario: varianza + trend): stessa
        # finestra di stagioni_prec usata per ma3 (qui back=0,1,2, cioe'
        # N-1,N-2,N-3 rispetto alla stagione target 2026-27 - coerente col
        # builder che usa back=1,2,3 rispetto alla stagione target storica,
        # la differenza di indice e' dovuta a come stagioni_prec e'
        # popolato sopra in questo script, non un errore). Invertito per il
        # trend (serve ordine cronologico "piu' vecchio prima").
        stagioni_prec_cronologiche = list(reversed(stagioni_prec))
        for m in METRICHE_BASE:
            riga[f"{m}_std3"] = std_or_none([s[m] for s in stagioni_prec])
            riga[f"{m}_trend3"] = trend_or_none([s[m] for s in stagioni_prec_cronologiche])

        stagioni_tutte_precedenti = [
            per_player[player_id][s] for s in stagioni_giocatore if STAGIONE_IDX[s] <= idx_n1
        ]
        for m in METRICHE_BASE:
            riga[f"{m}_career_mean"] = mean_or_none([s[m] for s in stagioni_tutte_precedenti])

        riga["eta_n1"] = eta_map.get((player_id, STAGIONE_N1))

        riga["squadra_in_champions_n1"] = 1.0 if agg_n1["squadra_giocatore"] in SQUADRE_CHAMPIONS_N1 else 0.0
        riga["squadra_in_champions_target"] = 1.0 if squadra_nome_target in SQUADRE_CHAMPIONS_TARGET else 0.0

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

        # forum Gruppo Esperti: qui TUTTE le righe sono 2026-27 per
        # costruzione (questo script prevede solo la stagione target),
        # quindi nessun filtro per stagione necessario (a differenza del
        # builder storico) - solo lookup diretto per player_id, None se
        # non matchato/non presente.
        forum = forum_map.get(player_id)
        riga["titolarita_forum"] = forum["titolarita_forum"] if forum else None
        riga["media_voto_forum"] = forum["media_voto_forum"] if forum else None
        riga["salute_forum"] = forum["salute_forum"] if forum else None
        riga["bonus_forum"] = forum["bonus_forum"] if forum else None
        riga["consiglio_esperti_forum"] = forum["consiglio_esperti_forum"] if forum else None
        riga["totale_forum"] = forum["totale_forum"] if forum else None

        ctx_n1 = classifica_finale.get((STAGIONE_N1, agg_n1["squadra_giocatore"]))
        riga["squadra_punti_finali_n1"] = ctx_n1["punti_pre"] if ctx_n1 else None
        riga["squadra_posizione_finale_n1"] = ctx_n1["posizione_pre"] if ctx_n1 else None
        if cambio_squadra == 1.0:
            ctx_new = classifica_finale.get((STAGIONE_N1, squadra_nome_target))
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

        X = pd.DataFrame([riga])[FEATURE_ORDER]
        for c in X.columns:
            if c not in CAT_COLS:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in CAT_COLS:
            X[c] = X[c].astype("category")

        pred = {}
        for model_file, out_name in TARGETS.items():
            pred[out_name] = float(models[model_file].predict(X)[0])

        # LightGBM regression non e' vincolata a valori non-negativi: per i
        # conteggi che per definizione non possono essere negativi (gol,
        # assist, presenze) una predizione vicina a 0 puo' uscire leggermente
        # sotto zero (bug scoperto dalla test suite: presenze_titolare_previste
        # osservato a -0.2 sui dati reali). Clampiamo a 0 solo queste 4
        # metriche - fantamedia/voto/bonus_netti NON vengono clampati perche'
        # bonus_netti puo' essere legittimamente negativo (piu' malus che
        # bonus) e fantamedia/voto non hanno mai mostrato il problema.
        for count_key in ("gol", "assist", "presenze_previste", "presenze_titolare_previste"):
            pred[count_key] = max(pred[count_key], 0.0)

        risultati.append({
            "squadra": squadra_sigla,
            "nome": agg_n1["nome_giocatore"],
            "ruolo": q_target["ruolo_classic"],
            "quotazione_iniziale_2026_27": q_target["quotazione_iniziale"],
            "presenze_2025_26": int(agg_n1["presenze"]),
            "pred_fantamedia": round(pred["fantamedia"], 2),
            "pred_voto": round(pred["voto"], 2),
            "pred_gol": round(pred["gol"], 1),
            "pred_assist": round(pred["assist"], 1),
            "pred_bonus_netti": round(pred["bonus_netti"], 1),
            "pred_presenze_previste": round(pred["presenze_previste"], 1),
            "pred_presenze_titolare_previste": round(pred["presenze_titolare_previste"], 1),
            "titolarita_forum_esperti": riga["titolarita_forum"],
            "salute_forum_esperti": riga["salute_forum"],
            "consiglio_forum_esperti": riga["consiglio_esperti_forum"],
            "totale_forum_esperti": riga["totale_forum"],
        })

    risultati.sort(key=lambda r: (r["squadra"], -(r["quotazione_iniziale_2026_27"] or 0)))

    if not risultati:
        log.warning("Nessun giocatore previsto: lista risultati vuota, nessun CSV scritto.")
        return

    fieldnames = list(risultati[0].keys())
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(risultati)

    tot_non_prevedibili = sum(len(v) for v in non_prevedibili_per_squadra.values())
    log.info("=== Previsioni Serie A 2026-27: %d giocatori PREVISTI su %d totali (%.1f%%) ===",
              len(risultati), len(tutti_2026_27), 100 * len(risultati) / len(tutti_2026_27))
    log.info("Output CSV: %s", OUT_PATH)
    log.info("")
    log.info("Giocatori 2026-27 SENZA storico Serie A 2025-26 (non prevedibili, %d totali):", tot_non_prevedibili)
    for squadra in sorted(non_prevedibili_per_squadra):
        nomi = non_prevedibili_per_squadra[squadra]
        log.info("  %s (%d): %s", squadra, len(nomi), nomi)


if __name__ == "__main__":
    main()
