#!/usr/bin/env python3
"""
Training e validazione di un modello ML che predice il "voto" (valutazione
tecnica di base assegnata dalla redazione) di un giocatore in una partita,
combinando statistiche individuali di partita (Understat), contesto squadra
di stagione (Lega Serie A), modulo tattico, allenatore (dove disponibile) e
classifica dinamica pre-partita (squadra e avversario).

DECISIONE DI TARGET: si usa "voto" (giudizio tecnico base) e NON
"fantavoto". fantavoto = voto + bonus/malus (gol*3, assist*1, ammonizione
-0.5, espulsione -1, rigore parato +3, rigore sbagliato -3, autogol -2,
gol_subiti portiere -1, mvp). Le colonne bonus (gol_fatti, assist, ecc.)
sono già feature del dataset: se il target fosse fantavoto, il modello
imparerebbe in gran parte una funzione quasi deterministica delle sue
stesse feature (leakage circolare), rendendo la validazione poco
significativa. "voto" è invece un giudizio più propriamente "di
prestazione" che le feature di questo dataset (xG, xA, tiri, contesto
squadra/avversario/modulo) possono legittimamente aiutare a spiegare/
predire senza essere una loro funzione deterministica.

FONTE VOTO: si usa solo fonte_voto == "redazione" (la fonte più completa
e usata come riferimento principale su fantacalcio.it) per evitare di
mescolare nello stesso training set 3 giudizi potenzialmente diversi
sulla stessa prestazione (redazione/statistico/italia), che aggiungerebbe
rumore/ambiguità non necessaria in questa prima versione del modello.

FILTRO: si escludono le righe con senza_voto=True (S.V. - giocatore non
valutabile, tipicamente subentrati con troppi pochi minuti) perché non
hanno un target valido.

SPLIT TEMPORALE (per evitare data leakage temporale - il modello non deve
mai vedere durante il training informazioni "dal futuro" rispetto a
quello che vede in validazione/test):
  - train:      stagioni 2015-16 .. 2022-23
  - validation: stagione  2023-24
  - test:       stagioni 2024-25, 2025-26

FEATURE escluse esplicitamente: identificatori (match_id, player_id,
nome_giocatore, data), il target stesso e le sue varianti (fantavoto,
senza_voto), fonte_voto (fissa per costruzione in questo script).

Output:
  work/data/model_train_log.txt (metriche, feature importance, bias check)
  work/models/lgbm_voto_v1.txt (modello LightGBM salvato)

Uso:
  python3 train_model.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_PATH = DATA_DIR / "feature_dataset_v2.csv"
LOG_PATH = DATA_DIR / "model_train_log.txt"
MODEL_PATH = MODELS_DIR / "lgbm_voto_v1.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("train_model")

TRAIN_SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                  "2020-21", "2021-22", "2022-23"]
VAL_SEASONS = ["2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

TARGET_COL = "voto"

# Colonne identificative/non-feature da escludere sempre
ID_COLS = [
    "match_id", "data", "nome_giocatore", "player_id", "fonte_voto",
    "senza_voto", "voto", "fantavoto",
]

CATEGORICAL_COLS = [
    "ruolo", "squadra_giocatore", "squadra_casa", "squadra_ospite",
    "squadra_modulo", "avversario_modulo", "squadra_allenatore", "avversario_allenatore",
    "understat_position",
]

# "stagione" viene usata solo per lo split temporale, non come feature: le
# stagioni di test (2024-25, 2025-26) sono categorie mai viste nel train,
# quindi non avrebbe senso trattarla come categorica di input (il modello
# deve generalizzare a stagioni future in base al contesto, non "imparare"
# l'etichetta della stagione).
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
    stats = stats[stats["count"] >= 30]  # solo gruppi con numerosità minima
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

    log.info("Inizio training LightGBM...")
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

    # --- Valutazione finale ---
    pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    valuta(y_train, pred_train, "TRAIN")
    valuta(y_val, pred_val, "VALIDATION")
    valuta(y_test, pred_test, "TEST")

    # baseline naive: predire sempre la media del train
    baseline_pred_test = np.full(len(y_test), y_train.mean())
    valuta(y_test, baseline_pred_test, "TEST (baseline: media train)")

    # --- Feature importance ---
    importance = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols)
    importance = importance.sort_values(ascending=False)
    log.info("--- Top 20 feature per importanza (gain) ---\n%s", importance.head(20).to_string())

    # --- Bias check su test set ---
    df_test = df[test_mask]
    for group_col in ["squadra_giocatore", "ruolo", "squadra_modulo", "squadra_allenatore"]:
        if group_col in df_test.columns:
            bias_check(df_test, y_test, pred_test, group_col, "TEST")

    log.info("Analisi completata.")


if __name__ == "__main__":
    main()
