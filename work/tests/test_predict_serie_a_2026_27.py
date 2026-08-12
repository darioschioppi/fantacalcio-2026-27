"""Test funzionali per predict_serie_a_2026_27.py.

Il test piu' importante e' quello su FEATURE_ORDER: il modulo passa a
LightGBM Booster.predict() un array POSIZIONALE (non un DataFrame con nomi
colonna), quindi FEATURE_ORDER (hardcoded in predict_serie_a_2026_27.py) deve
coincidere esattamente, elemento per elemento, con l'ordine delle feature
usato in training (prepara_feature_cols() in train_model_rendimento_stagionale.py,
che a sua volta riflette l'ordine delle colonne nel CSV scritto da
build_stagione_giocatore_dataset.py). Il codice stesso segnala questo rischio
in un commento ("ATTENZIONE ORDINE...") ma non era testato: se in futuro
build_stagione_giocatore_dataset.py cambia l'ordine/aggiunge colonne senza
aggiornare FEATURE_ORDER, le predizioni sballerebbero silenziosamente (nessun
errore, solo numeri sbagliati) - questo test lo intercetterebbe.
"""
import csv

import pytest

from conftest import import_fresh


@pytest.fixture
def predict_module_result(build_dataset_module, pipeline_dirs):
    train_module = import_fresh("train_model_rendimento_stagionale.py")
    train_module.main()
    predict_module = import_fresh("predict_serie_a_2026_27.py")
    predict_module.main()
    return predict_module, pipeline_dirs


def test_feature_order_coincide_con_ordine_training(build_dataset_module):
    """FEATURE_ORDER (predict) deve avere lo stesso insieme e lo stesso
    ordine delle colonne feature effettivamente presenti nel dataset di
    training (tutte le colonne tranne ID_COLS e ALL_TARGET_COLS)."""
    _, out_path = build_dataset_module
    with open(out_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    train_module = import_fresh("train_model_rendimento_stagionale.py")
    predict_module = import_fresh("predict_serie_a_2026_27.py")

    feature_cols_training = [
        c for c in header
        if c not in train_module.ID_COLS and c not in train_module.ALL_TARGET_COLS
    ]

    assert list(predict_module.FEATURE_ORDER) == feature_cols_training, (
        "FEATURE_ORDER in predict_serie_a_2026_27.py e' DIVERSO dall'ordine "
        "feature usato in training: le predizioni sarebbero silenziosamente "
        "errate (LightGBM Booster.predict() usa l'ordine posizionale)."
    )


def test_predict_produce_output_con_range_plausibili(predict_module_result):
    _, dirs = predict_module_result
    out_path = dirs["data_dir"] / "previsioni_serie_a_2026_27.csv"
    assert out_path.exists()
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    for row in rows:
        assert 0.0 <= float(row["pred_fantamedia"]) <= 10.0
        assert 0.0 <= float(row["pred_voto"]) <= 10.0
        assert float(row["pred_gol"]) >= 0.0
        assert float(row["pred_assist"]) >= 0.0
        assert 0.0 <= float(row["pred_presenze_previste"]) <= 38.0
        assert 0.0 <= float(row["pred_presenze_titolare_previste"]) <= 38.0


def test_lista_risultati_vuota_non_solleva_indexerror(build_dataset_module, pipeline_dirs):
    """Regressione end-to-end per il fix difensivo: se il CSV quotazioni non
    ha nessuna riga stagione=='2026-27' (quindi tutti_2026_27 e' vuoto),
    main() non deve piu' sollevare IndexError su risultati[0].keys() - deve
    loggare un warning e uscire senza scrivere il CSV di output."""
    train_module = import_fresh("train_model_rendimento_stagionale.py")
    train_module.main()

    # rimuove tutte le righe 2026-27 dal file quotazioni nella dir fixture,
    # cosi' tutti_2026_27 risulta vuoto in predict_serie_a_2026_27.main()
    quot_path = pipeline_dirs["data_dir"] / "quotazioni_fantacalcio_storico_2015_2026.csv"
    with open(quot_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if r["stagione"] != "2026-27"]
    with open(quot_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    predict_module = import_fresh("predict_serie_a_2026_27.py")
    # OUT_PATH puo' gia' esistere: pipeline_dirs pre-popola
    # previsioni_serie_a_2026_27.csv dalla fixture (serve ai test fuzzy che
    # non ri-eseguono predict). Qui verifichiamo che main(), con lista
    # risultati vuota, NON lo sovrascriva (e non sollevi IndexError) -
    # quindi partiamo rimuovendolo esplicitamente.
    if predict_module.OUT_PATH.exists():
        predict_module.OUT_PATH.unlink()

    predict_module.main()  # non deve solzare IndexError

    assert not predict_module.OUT_PATH.exists()
