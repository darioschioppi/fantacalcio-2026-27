"""Test funzionali per build_stagione_giocatore_dataset.py sulle fixture reali."""
import csv

import pytest


def test_output_csv_creato(build_dataset_module):
    _, out_path = build_dataset_module
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_118_colonne(build_dataset_module):
    _, out_path = build_dataset_module
    with open(out_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert len(header) == 118


def test_nessun_nan_nelle_colonne_target_per_righe_valide(build_dataset_module):
    """I target sono calcolati da un'aggregazione (aggrega_per_giocatore_stagione)
    che esiste solo se il giocatore ha presenze valide in quella stagione target:
    quindi per ogni riga scritta i target non devono mai essere vuoti."""
    _, out_path = build_dataset_module
    target_cols = ["fantamedia_target", "gol_target", "assist_target",
                   "bonus_netti_target", "presenze_target", "voto_medio_target",
                   "presenze_titolare_target"]
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    for row in rows:
        for col in target_cols:
            assert row[col] not in (None, ""), f"{col} vuoto in riga player_id={row['player_id']} stagione={row['stagione_target']}"


def test_stagione_target_in_range_storico(build_dataset_module):
    _, out_path = build_dataset_module
    stagioni_valide = {"2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
                        "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"}
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["stagione_target"] in stagioni_valide


def test_presenze_target_non_negative(build_dataset_module):
    _, out_path = build_dataset_module
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert float(row["presenze_target"]) >= 0
