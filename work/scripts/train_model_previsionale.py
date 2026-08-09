#!/usr/bin/env python3
"""
Training e validazione del modello PREVISIONALE (predice il "voto" usando
SOLO informazione disponibile PRIMA del fischio d'inizio), su
feature_dataset_v3.csv (vedi build_feature_dataset_v3.py per la
costruzione delle feature e la rimozione esplicita di tutte le colonne di
risultato/bonus della partita corrente che erano invece presenti in v1/v2).

Stessa struttura di train_model.py (target, filtro fonte_voto, split
temporale, LightGBM, bias check) - le uniche differenze sono il file di
input (v3 invece di v2) e l'elenco di feature (niente colonne understat_*
di partita corrente, niente gol_fatti/assist/ecc., niente squadra_total-*
di fine stagione: sostituite da forma dinamica rolling e lag giocatore).

MAE atteso PIÙ ALTO di v1 (0.3133 su test) perché il problema previsionale
è oggettivamente più difficile (non si "ricostruisce" il voto da un
risultato già noto, si prova a indovinarlo prima) - è il risultato onesto
da riportare, non un bug o una regressione da correggere.

Output:
  work/data/model_train_previsionale_log.txt
  work/models/lgbm_voto_previsionale_v1.txt

Uso:
  python3 train_model_previsionale.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_PATH = DATA_DIR / "feature_dataset_v3.csv"
LOG_PATH = DATA_DIR / "model_train_previsionale_log.txt"
MODEL_PATH = MODELS_DIR / "lgbm_voto_previsionale_v1.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("train_model_previsionale")

TRAIN_SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                  "2020-21", "2021-22", "2022-23"]
VAL_SEASONS = ["2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

TARGET_COL = "voto"

ID_COLS = [
    "match_id", "data", "nome_giocatore", "player_id", "fonte_voto",
    "senza_voto", "voto", "fantavoto",
]

CATEGORICAL_COLS = [
    "ruolo", "squadra_giocatore", "squadra_casa", "squadra_ospite",
    "squadra_modulo", "avversario_modulo", "squadra_allenatore", "avversario_allenatore",
]

NON_FEATURE_EXTRA = ["stagione"]


def carica_dataset():
    log.info("Caricamento %s ...", FEATURE_PATH)
    df = pd.read_csv(FEATURE_PATH, low_memory=False)
    log.info("Righe totali caricate: %d", len(df))

    df = df[df["fonte_voto"] == "redazione"].copy()
    log.info("Dopo filtro fonte_voto=redazione: %d righe", len(df))

    df = df[df["senza_voto"].astype(str) != "True"].copy()
    df = df.dropna(subset=[TARGET_COL])
    log.info("Dopo filtro senza_voto/target nullo: %d righe", len(df))

    return df


def prepara_feature(df):
    feature_cols = [c for c in df.columns if c not in ID_COLS and c not in NON_FEATURE_EXTRA and c != "giornata"]
    feature_cols = ["giornata"] + feature_cols

    X = df[feature_cols].copy()
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")

    y = df[TARGET_COL].astype(float)
    return X, y, feature_cols


def split_temporale(df):
    train_mask = df["stagione"].isin(TRAIN_SEASONS)
    val_mask = df["stagione"].isin(VAL_SEASONS)
    test_mask = df["stagione"].isin(TEST_SEASONS)
    log.info("Train: %d righe (%s)", train_mask.sum(), TRAIN_SEASONS)
    log.info("Validation: %d righe (%s)", val_mask.sum(), VAL_SEASONS)
    log.info("Test: %d righe (%s)", test_mask.sum(), TEST_SEASONS)
    return train_mask, val_mask, test_mask


def valuta(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    log.info("[%s] MAE=%.4f RMSE=%.4f R2=%.4f (n=%d)", label, mae, rmse, r2, len(y_true))
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(y_true)}


def bias_check(df_subset, y_true, y_pred, group_col, label):
    tmp = df_subset[[group_col]].copy()
    tmp["err"] = y_pred - y_true.values
    stats = tmp.groupby(group_col)["err"].agg(["mean", "count"]).sort_values("mean")
    stats = stats[stats["count"] >= 30]
    log.info("--- Bias check per %s (%s) - top 5 sovrastima / sottostima ---", group_col, label)
    log.info("Sottostimati (errore medio più negativo):\n%s", stats.head(5).to_string())
    log.info("Sovrastimati (errore medio più positivo):\n%s", stats.tail(5).to_string())
    return stats


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = carica_dataset()
    train_mask, val_mask, test_mask = split_temporale(df)

    X, y, feature_cols = prepara_feature(df)
    log.info("Feature usate (%d): %s", len(feature_cols), feature_cols)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_set, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    log.info("Inizio training LightGBM (modello previsionale)...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
    )
    log.info("Training completato. Best iteration: %d", model.best_iteration)

    model.save_model(str(MODEL_PATH), num_iteration=model.best_iteration)
    log.info("Modello salvato in %s", MODEL_PATH)

    pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    valuta(y_train, pred_train, "TRAIN")
    valuta(y_val, pred_val, "VALIDATION")
    valuta(y_test, pred_test, "TEST")

    baseline_pred_test = np.full(len(y_test), y_train.mean())
    valuta(y_test, baseline_pred_test, "TEST (baseline: media train)")

    # baseline aggiuntiva specifica per il problema previsionale: predire
    # sempre il lag medio del giocatore stesso (se disponibile, altrimenti
    # la media train) - confronto più severo della semplice media globale,
    # perché usa già una parte dell'informazione "pre-partita" disponibile.
    if "voto_lag_mean_5" in df.columns:
        lag_test = df.loc[test_mask, "voto_lag_mean_5"].fillna(y_train.mean())
        valuta(y_test, lag_test.values, "TEST (baseline: voto_lag_mean_5 del giocatore)")

    importance = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols)
    importance = importance.sort_values(ascending=False)
    log.info("--- Top 20 feature per importanza (gain) ---\n%s", importance.head(20).to_string())

    df_test = df[test_mask]
    for group_col in ["squadra_giocatore", "ruolo", "squadra_modulo", "squadra_allenatore"]:
        if group_col in df_test.columns:
            bias_check(df_test, y_test, pred_test, group_col, "TEST")

    log.info("Analisi completata.")


if __name__ == "__main__":
    main()
