#!/usr/bin/env python3
"""
Script ONE-OFF (non parte della suite pytest) per estrarre fixture di test
reali da 18 player_id con storico continuo su almeno 6 stagioni (incluse
2025-26 e 2026-27), propagando il filtro a tutti i CSV di input della
pipeline. Le fixture preservano formati ed edge case reali (cambi squadra,
mancato match Understat, dati anagrafici assenti) invece di dati sintetici.

Uso (dalla root del repo):
  python3 work/tests/_extract_fixtures.py
Scrive in work/tests/fixtures/*.csv
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 18 player_id con quotazione in tutte le stagioni 2015-16..2026-27 (>=6 stagioni
# comprese 2025-26 e 2026-27), scelti da quotazioni_fantacalcio_storico_2015_2026.csv.
IDS = [4, 133, 152, 218, 252, 294, 309, 327, 333, 460, 487, 530, 536, 543, 608, 779, 787, 827]
IDS_STR = [str(i) for i in IDS]

def main():
    # --- voti_storici ---
    voti = pd.read_csv(DATA_DIR / "voti_storici_2015_2026.csv", dtype=str)
    voti_sub = voti[voti.player_id.astype(str).isin(IDS_STR)].copy()
    voti_sub.to_csv(OUT_DIR / "voti_storici_sample.csv", index=False)
    print(f"voti_storici_sample: {len(voti_sub)} righe")

    # --- quotazioni ---
    quot = pd.read_csv(DATA_DIR / "quotazioni_fantacalcio_storico_2015_2026.csv", dtype=str)
    quot_sub = quot[quot.player_id.astype(str).isin(IDS_STR)].copy()
    quot_sub.to_csv(OUT_DIR / "quotazioni_sample.csv", index=False)
    print(f"quotazioni_sample: {len(quot_sub)} righe")

    # --- player_name_mapping: bridge (stagione, squadra, nome_fantacalcio) -> player_id_understat ---
    key_set = set(zip(voti_sub.stagione, voti_sub.squadra_giocatore, voti_sub.nome_giocatore))
    mapping = pd.read_csv(DATA_DIR / "player_name_mapping.csv", dtype=str)
    mapping_sub = mapping[mapping.apply(
        lambda r: (r.stagione, r.squadra, r.nome_fantacalcio) in key_set, axis=1)].copy()
    mapping_sub.to_csv(OUT_DIR / "player_name_mapping_sample.csv", index=False)
    print(f"player_name_mapping_sample: {len(mapping_sub)} righe")

    understat_ids = set(mapping_sub.player_id_understat.dropna().unique())

    # --- understat: filtrato per player_id_understat risolti dal mapping ---
    understat = pd.read_csv(DATA_DIR / "understat_player_match_stats_storico_2015_2026.csv", dtype=str)
    understat_sub = understat[understat.player_id.astype(str).isin(understat_ids)].copy()
    understat_sub.to_csv(OUT_DIR / "understat_sample.csv", index=False)
    print(f"understat_sample: {len(understat_sub)} righe")

    # --- classifica_dinamica: filtrata per le squadre coinvolte ---
    squadre = set(voti_sub.squadra_giocatore.unique())
    classifica = pd.read_csv(DATA_DIR / "classifica_dinamica_storico_2015_2026.csv", dtype=str)
    classifica_sub = classifica[classifica.squadra.isin(squadre)].copy()
    classifica_sub.to_csv(OUT_DIR / "classifica_dinamica_sample.csv", index=False)
    print(f"classifica_dinamica_sample: {len(classifica_sub)} righe")

    # --- eta_giocatori ---
    eta = pd.read_csv(DATA_DIR / "eta_giocatori_storico_2015_2026.csv", dtype=str)
    eta_sub = eta[eta.player_id.astype(str).isin(IDS_STR)].copy()
    eta_sub.to_csv(OUT_DIR / "eta_sample.csv", index=False)
    print(f"eta_sample: {len(eta_sub)} righe")

    # --- infortuni ---
    infortuni = pd.read_csv(DATA_DIR / "infortuni_giocatori_storico_2015_2026.csv", dtype=str)
    infortuni_sub = infortuni[infortuni.player_id.astype(str).isin(IDS_STR)].copy()
    infortuni_sub.to_csv(OUT_DIR / "infortuni_sample.csv", index=False)
    print(f"infortuni_sample: {len(infortuni_sub)} righe")

    # --- profilo_giocatori ---
    profilo = pd.read_csv(DATA_DIR / "profilo_giocatori_storico_2015_2026.csv", dtype=str)
    profilo_sub = profilo[profilo.player_id.astype(str).isin(IDS_STR)].copy()
    profilo_sub.to_csv(OUT_DIR / "profilo_sample.csv", index=False)
    print(f"profilo_sample: {len(profilo_sub)} righe")

    # --- forum_esperti_pagelle_2026_27 ---
    forum = pd.read_csv(DATA_DIR / "forum_esperti_pagelle_2026_27.csv", dtype=str)
    forum_sub = forum[forum.player_id.astype(str).isin(IDS_STR)].copy()
    forum_sub.to_csv(OUT_DIR / "forum_esperti_sample.csv", index=False)
    print(f"forum_esperti_sample: {len(forum_sub)} righe")

    # --- previsioni_serie_a_2026_27 (per test fuzzy_sorprese_forum, indipendente da IDS) ---
    previsioni = pd.read_csv(DATA_DIR / "previsioni_serie_a_2026_27.csv", dtype=str)
    nomi_sub = set(quot_sub[quot_sub.stagione == "2026-27"].nome_giocatore.unique())
    previsioni_sub = previsioni[previsioni.nome.isin(nomi_sub)].copy()
    if previsioni_sub.empty:
        previsioni_sub = previsioni.head(15).copy()
    previsioni_sub.to_csv(OUT_DIR / "previsioni_sample.csv", index=False)
    print(f"previsioni_sample: {len(previsioni_sub)} righe")


if __name__ == "__main__":
    main()
