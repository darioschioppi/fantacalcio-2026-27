#!/usr/bin/env python3
"""
Training e validazione di 4 modelli LightGBM che predicono il RENDIMENTO
STAGIONALE AGGREGATO di un giocatore (non il voto di una singola partita),
usando SOLO informazione nota prima dell'inizio della stagione target -
vedi build_stagione_giocatore_dataset.py per la costruzione del dataset e
il principio anti-leakage (feature calcolate solo su stagioni <= N-1, più
la quotazione ufficiale iniziale della stagione N stessa, pubblicata prima
del campionato quindi non leakage).

I 4 target, tutti richiesti esplicitamente da Dario ("fantamedia redazione
fantacalcio di ogni singolo giocatore e anche gol, assist e bonus"):
  - fantamedia_target   (media fantavoto sulla stagione)
  - gol_target          (somma gol fatti sulla stagione)
  - assist_target       (somma assist sulla stagione)
  - bonus_netti_target  (somma fantavoto-voto sulla stagione, bonus/malus
                          netti già calcolati dalla redazione)

Si allena un modello LightGBM SEPARATO per ciascun target (stessa lista di
feature per tutti e 4, cambia solo il target). Per ognuno si confrontano
TRE baseline, non solo una:
  1. "media train" (predici sempre la media del target sul train) - la
     baseline più debole, incluso solo per continuità con le fasi precedenti.
  2. "ripeti anno precedente" (predici il valore lag1 dello stesso target,
     es. fantamedia_target ~= fantamedia_lag1) - baseline autoregressiva
     naive, più severa e realistica: il modello deve batterla per
     dimostrare che aggiunge valore rispetto a "guarda solo l'anno scorso".
  3. "quotazione ufficiale" (regressione lineare univariata,
     fittata SOLO sul train, che usa quotazione_iniziale_target come unico
     predittore) - rappresenta "quanto già sa il prezzo d'asta ufficiale
     dato dagli esperti"; il modello deve aggiungere valore SOPRA questa
     baseline per giustificare la sua complessità.

Split temporale (a livello di stagione_target, coerente con le fasi
precedenti): train = 2017-18..2022-23, val = 2023-24, test = 2024-25/2025-26.
(2016-17 esiste nel dataset ma è esclusa da train: è la prima stagione con
feature lag disponibili, usata solo per costruire lo storico, non incisa
qui per evitare un anno con feature ma3 quasi sempre vuote).

Output:
  work/data/model_train_stagionale_log.txt
  work/models/lgbm_{target}_stagionale_v1.txt   (x4)

Uso:
  python3 train_model_rendimento_stagionale.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_PATH = DATA_DIR / "stagione_giocatore_dataset_2015_2026.csv"
LOG_PATH = DATA_DIR / "model_train_stagionale_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("train_model_rendimento_stagionale")

TRAIN_SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
VAL_SEASONS = ["2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

TARGETS = ["fantamedia_target", "gol_target", "assist_target", "bonus_netti_target"]
TARGET_TO_LAG1 = {
    "fantamedia_target": "fantamedia_lag1",
    "gol_target": "gol_lag1",
    "assist_target": "assist_lag1",
    "bonus_netti_target": "bonus_netti_lag1",
}

ID_COLS = ["stagione_target", "player_id", "nome_giocatore",
           "squadra_giocatore_target", "squadra_giocatore_n1"]
ALL_TARGET_COLS = ["fantamedia_target", "gol_target", "assist_target",
                    "bonus_netti_target", "presenze_target"]
CATEGORICAL_COLS = ["ruolo", "nazionalita", "piede_dominante"]
QUOTAZIONE_COL = "quotazione_iniziale_target"


def carica_dataset():
    log.info("Caricamento %s ...", FEATURE_PATH)
    df = pd.read_csv(FEATURE_PATH, low_memory=False)
    log.info("Righe totali caricate: %d", len(df))
    return df


def split_temporale(df):
    train_mask = df["stagione_target"].isin(TRAIN_SEASONS)
    val_mask = df["stagione_target"].isin(VAL_SEASONS)
    test_mask = df["stagione_target"].isin(TEST_SEASONS)
    log.info("Train: %d righe (%s)", train_mask.sum(), TRAIN_SEASONS)
    log.info("Validation: %d righe (%s)", val_mask.sum(), VAL_SEASONS)
    log.info("Test: %d righe (%s)", test_mask.sum(), TEST_SEASONS)
    return train_mask, val_mask, test_mask


def prepara_feature_cols(df):
    feature_cols = [c for c in df.columns if c not in ID_COLS and c not in ALL_TARGET_COLS]
    return feature_cols


def valuta(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    log.info("  [%s] MAE=%.4f RMSE=%.4f R2=%.4f (n=%d)", label, mae, rmse, r2, len(y_true))
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(y_true)}


def baseline_quotazione(df, train_mask, test_mask, target_col):
    """Regressione lineare univariata quotazione_iniziale_target -> target,
    fittata SOLO sul train, valutata su test. Righe con quotazione mancante
    escluse dal fit E dalla valutazione di questa baseline specifica
    (riportato il numero escluso, non finto riempimento)."""
    train_ok = train_mask & df[QUOTAZIONE_COL].notna() & df[target_col].notna()
    test_ok = test_mask & df[QUOTAZIONE_COL].notna() & df[target_col].notna()
    n_escluse = int(test_mask.sum() - test_ok.sum())

    X_train = df.loc[train_ok, [QUOTAZIONE_COL]].values
    y_train = df.loc[train_ok, target_col].values
    X_test = df.loc[test_ok, [QUOTAZIONE_COL]].values
    y_test = df.loc[test_ok, target_col].values

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    pred_test = reg.predict(X_test)

    log.info("  (baseline quotazione: %d righe test escluse per quotazione/target mancante)", n_escluse)
    return valuta(y_test, pred_test, "TEST (baseline: quotazione_iniziale_target, regressione lineare)")


def allena_target(df, train_mask, val_mask, test_mask, target_col, feature_cols, cat_features):
    log.info("=== TARGET: %s ===", target_col)

    valid_mask = df[target_col].notna()
    tr = train_mask & valid_mask
    va = val_mask & valid_mask
    te = test_mask & valid_mask

    X = df[feature_cols].copy()
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df[target_col].astype(float)

    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    X_test, y_test = X[te], y[te]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_set, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 15,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)],
    )
    log.info("  Training completato. Best iteration: %d", model.best_iteration)

    model_path = MODELS_DIR / f"lgbm_{target_col.replace('_target', '')}_stagionale_v1.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    log.info("  Modello salvato in %s", model_path)

    pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    valuta(y_train, pred_train, "TRAIN")
    valuta(y_val, pred_val, "VALIDATION")
    res_test = valuta(y_test, pred_test, "TEST")

    # baseline 1: media train
    baseline_media = np.full(len(y_test), y_train.mean())
    valuta(y_test, baseline_media, "TEST (baseline: media train)")

    # baseline 2: ripeti anno precedente (lag1 dello stesso target)
    lag1_col = TARGET_TO_LAG1[target_col]
    lag1_test = df.loc[te, lag1_col]
    mask_lag1_ok = lag1_test.notna()
    if mask_lag1_ok.sum() > 0:
        valuta(y_test[mask_lag1_ok.values], lag1_test[mask_lag1_ok].values,
               f"TEST (baseline: ripeti {lag1_col})")
    else:
        log.warning("  Nessuna riga test con %s disponibile per la baseline 'ripeti anno precedente'", lag1_col)

    # baseline 3: quotazione ufficiale (regressione lineare univariata)
    baseline_quotazione(df, train_mask & valid_mask, test_mask & valid_mask, target_col)

    # feature importance
    importance = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols)
    importance = importance.sort_values(ascending=False)
    log.info("  --- Top 10 feature per importanza (gain) ---\n%s", importance.head(10).to_string())

    return res_test


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = carica_dataset()
    train_mask, val_mask, test_mask = split_temporale(df)
    feature_cols = prepara_feature_cols(df)
    log.info("Feature usate (%d): %s", len(feature_cols), feature_cols)
    cat_features = [c for c in CATEGORICAL_COLS if c in feature_cols]

    risultati = {}
    for target_col in TARGETS:
        risultati[target_col] = allena_target(df, train_mask, val_mask, test_mask, target_col, feature_cols, cat_features)

    log.info("=== RIEPILOGO FINALE (MAE test per target) ===")
    for target_col, res in risultati.items():
        log.info("  %s: MAE=%.4f R2=%.4f", target_col, res["mae"], res["r2"])

    log.info("Analisi completata.")


if __name__ == "__main__":
    main()
