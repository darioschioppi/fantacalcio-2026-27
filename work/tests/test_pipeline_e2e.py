"""Test end-to-end: esegue in sequenza build -> train -> predict -> fuzzy
sulle fixture reali in una working dir temporanea isolata, e verifica che
l'output finale abbia le colonne attese senza righe anomale. E' il test che
copre l'intera catena, complementare (non sostitutivo) ai test unitari per
componente negli altri file."""
import csv

from conftest import import_fresh

COLONNE_FINALI_ATTESE = {
    "squadra", "nome", "ruolo", "quotazione_iniziale_2026_27", "presenze_2025_26",
    "pred_fantamedia", "pred_voto", "pred_gol", "pred_assist", "pred_bonus_netti",
    "pred_presenze_previste", "pred_presenze_titolare_previste",
    "titolarita_forum_esperti", "salute_forum_esperti", "consiglio_forum_esperti",
    "totale_forum_esperti", "indice_sorpresa", "categoria_sorpresa",
}


def test_pipeline_completa_build_train_predict_fuzzy(pipeline_env, pipeline_dirs):
    build_module = import_fresh("build_stagione_giocatore_dataset.py")
    build_module.main()
    assert build_module.OUT_PATH.exists()

    train_module = import_fresh("train_model_rendimento_stagionale.py")
    risultati_train = train_module.main()
    assert len(risultati_train) == 7

    predict_module = import_fresh("predict_serie_a_2026_27.py")
    predict_module.main()
    assert predict_module.OUT_PATH.exists()

    fuzzy_module = import_fresh("fuzzy_sorprese_forum.py")
    fuzzy_module.main()
    assert fuzzy_module.OUT_PATH.exists()

    with open(fuzzy_module.OUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        colonne = set(reader.fieldnames)

    assert len(rows) > 0
    assert COLONNE_FINALI_ATTESE.issubset(colonne)

    for row in rows:
        # range plausibili sulle stesse metriche gia' testate in isolamento
        assert 0.0 <= float(row["pred_fantamedia"]) <= 10.0
        assert 0.0 <= float(row["pred_voto"]) <= 10.0
        if row["indice_sorpresa"] not in ("", None):
            assert -10.0 <= float(row["indice_sorpresa"]) <= 10.0
            assert row["categoria_sorpresa"] != ""
