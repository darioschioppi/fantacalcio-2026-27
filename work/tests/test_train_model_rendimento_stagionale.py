"""Test funzionali per train_model_rendimento_stagionale.py.

Test rapidi (default): training sulle fixture (18 player_id, 169 righe),
verificano che i 7 modelli vengano scritti, siano caricabili e producano
predizioni senza NaN, e che il training sia riproducibile (seed=42 fisso
nello script).

Test lento (@pytest.mark.slow): allena sul dataset REALE intero
(work/data/stagione_giocatore_dataset_2015_2026.csv, non le fixture) e
confronta MAE/R2 con i valori di riferimento noti (tolleranza +-15%
relativo) per intercettare regressioni silenziose introdotte da modifiche
successive a feature/training. Va eseguito manualmente dopo ogni modifica
a build_stagione_giocatore_dataset.py o train_model_rendimento_stagionale.py.
"""
import math

import lightgbm as lgb
import pytest

from conftest import import_fresh

TARGETS_ATTESI = ["fantamedia", "voto_medio", "gol", "assist",
                   "bonus_netti", "presenze", "presenze_titolare"]

# valori di riferimento noti (confermati via log reali dell'ultima run in
# work/data/model_train_stagionale_log.txt), usati SOLO dal test slow sul
# dataset reale intero.
METRICHE_RIFERIMENTO = {
    "fantamedia_target": {"mae": 0.3108, "r2": 0.5381},
    "voto_medio_target": {"mae": 0.1515, "r2": 0.2557},
    "gol_target": {"mae": 1.3191, "r2": 0.3708},
    "assist_target": {"mae": 1.0593, "r2": 0.2581},
    "bonus_netti_target": {"mae": 5.3560, "r2": 0.6039},
    "presenze_target": {"mae": 7.8641, "r2": 0.1762},
    "presenze_titolare_target": {"mae": 7.8421, "r2": 0.2855},
}
TOLLERANZA_RELATIVA = 0.15


@pytest.fixture
def train_module_result(build_dataset_module, pipeline_dirs):
    module = import_fresh("train_model_rendimento_stagionale.py")
    risultati = module.main()
    return module, risultati, pipeline_dirs["models_dir"]


def test_7_modelli_scritti_e_caricabili(train_module_result):
    _, _, models_dir = train_module_result
    for target in TARGETS_ATTESI:
        model_path = models_dir / f"lgbm_{target}_stagionale_v1.txt"
        assert model_path.exists(), f"modello mancante: {model_path}"
        booster = lgb.Booster(model_file=str(model_path))
        assert booster.num_trees() > 0


def test_nessun_nan_nelle_predizioni(train_module_result, pipeline_dirs):
    import pandas as pd
    module, _, models_dir = train_module_result
    df = pd.read_csv(module.FEATURE_PATH, low_memory=False)
    feature_cols = module.prepara_feature_cols(df)
    for c in module.CATEGORICAL_COLS:
        if c in feature_cols:
            df[c] = df[c].astype("category")
    for target in TARGETS_ATTESI:
        model_path = models_dir / f"lgbm_{target}_stagionale_v1.txt"
        booster = lgb.Booster(model_file=str(model_path))
        preds = booster.predict(df[feature_cols])
        assert not any(math.isnan(p) for p in preds), f"NaN nelle predizioni per {target}"


def test_riproducibilita_stesso_seed_stesse_predizioni(build_dataset_module, pipeline_dirs):
    """Il training usa seed=42 fisso (bagging/feature_fraction/lgb stocastico);
    due run consecutive sullo stesso dataset di fixture devono produrre lo
    stesso modello (stesse predizioni bit-per-bit su un sottoinsieme fisso)."""
    import pandas as pd

    module1 = import_fresh("train_model_rendimento_stagionale.py")
    module1.main()
    df = pd.read_csv(module1.FEATURE_PATH, low_memory=False)
    feature_cols = module1.prepara_feature_cols(df)
    for c in module1.CATEGORICAL_COLS:
        if c in feature_cols:
            df[c] = df[c].astype("category")
    model_path = pipeline_dirs["models_dir"] / "lgbm_fantamedia_stagionale_v1.txt"
    preds_run1 = lgb.Booster(model_file=str(model_path)).predict(df[feature_cols])

    module2 = import_fresh("train_model_rendimento_stagionale.py")
    module2.main()
    preds_run2 = lgb.Booster(model_file=str(model_path)).predict(df[feature_cols])

    assert list(preds_run1) == pytest.approx(list(preds_run2), abs=1e-9)


@pytest.mark.slow
def test_metriche_su_dataset_reale_entro_tolleranza():
    """Allena sui dati reali (nessuna env var override -> path di default
    work/data/) e confronta MAE/R2 test di ciascun target con i valori di
    riferimento noti, tolleranza +-15% relativo. Richiede che
    work/data/stagione_giocatore_dataset_2015_2026.csv esista gia' (prodotto
    dall'ultima run reale di build_stagione_giocatore_dataset.py)."""
    module = import_fresh("train_model_rendimento_stagionale.py")
    if not module.FEATURE_PATH.exists():
        pytest.skip(f"dataset reale non trovato: {module.FEATURE_PATH}")

    risultati = module.main()

    for target_col, riferimento in METRICHE_RIFERIMENTO.items():
        assert target_col in risultati, f"target {target_col} assente dai risultati"
        res = risultati[target_col]
        for metrica, val_rif in riferimento.items():
            val_attuale = res[metrica]
            tolleranza_abs = abs(val_rif) * TOLLERANZA_RELATIVA
            assert abs(val_attuale - val_rif) <= tolleranza_abs, (
                f"{target_col}.{metrica}: attuale={val_attuale:.4f} "
                f"riferimento={val_rif:.4f} tolleranza=+-{tolleranza_abs:.4f}"
            )
