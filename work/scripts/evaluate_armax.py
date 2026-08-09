#!/usr/bin/env python3
"""
Valutazione di fattibilità di ARMAX (AutoRegressive Moving Average with
eXogenous inputs, implementato in statsmodels come SARIMAX con order
stagionale nullo) come modello ALTERNATIVO/COMPLEMENTARE al gradient
boosting già allenato in train_model_previsionale.py.

VINCOLO CHIAVE (verificato su feature_dataset_v3.csv): la mediana di
presenze storiche per giocatore è 28 su tutto lo storico 2015-16..2025-26;
solo 388/2124 giocatori hanno >=100 presenze. Un ARIMA/ARMAX per-giocatore
richiede una serie ragionevolmente lunga e regolare per una stima stabile
dei parametri (p,d,q) - non è quindi sensato tentare un ARMAX per-giocatore
sull'intera popolazione. Si esegue perciò un ESPERIMENTO SCOPING mirato sul
sottoinsieme di giocatori con maggiore numerosità storica, non
un'integrazione totale nella pipeline.

SELEZIONE SOTTOINSIEME: giocatori con >=80 presenze valide (fonte_voto=
redazione, non senza_voto) nelle stagioni di TRAIN (2015-16..2022-23) e
>=8 presenze nelle stagioni di TEST (2024-25, 2025-26) - garantisce una
storia sufficiente per stimare il modello E un numero minimo di
osservazioni di test per un confronto MAE non rumoroso. Il filtro produce
~114 giocatori (verificato).

METODO:
  - endog = "voto" (stessa serie target del modello LightGBM previsionale)
  - exog = SOLO feature numeriche pre-partita già costruite per il
    modello previsionale (forma dinamica squadra/avversario, classifica
    pre, lag giocatore, presenze cumulate, giornata, indicatore casa/
    ospite) - le categoriche (modulo, allenatore, squadre) sono escluse
    perché SARIMAX non le gestisce nativamente in questa prima iterazione
    (nessuna one-hot: con storie di 60-280 osservazioni per giocatore,
    un one-hot con decine di categorie sovraparametrizzerebbe il modello).
  - valori NaN negli exog (es. primissime presenze senza lag disponibile,
    forma dinamica con n_partite=0) imputati con la MEDIANA GLOBALE di
    quella colonna sull'intero dataset di train - scelta semplice e
    trasparente, dichiarata esplicitamente (non un tentativo di "inventare"
    un valore realistico, solo evitare che SARIMAX fallisca sui NaN).
  - selezione ORDINE (p,d,q): piccola grid search { (1,0,0), (1,0,1),
    (2,0,1), (2,1,1) } scelta per AIC più basso sui dati di TRAIN+VAL del
    singolo giocatore (d=0 quasi sempre preferibile: il voto è una serie
    già stazionaria per costruzione, oscilla in un range fisso 1-10).
  - VALIDAZIONE WALK-FORWARD one-step-ahead: il modello viene stimato una
    volta sullo storico fino a fine 2023-24 (train+val), poi per ogni
    presenza di TEST (2024-25, 2025-26, in ordine cronologico) si
    aggiornano i dati osservati (statsmodels .append(..., refit=False):
    aggiorna lo stato del filtro di Kalman con l'osservazione reale senza
    ristimare i parametri - pratica standard per forecasting incrementale
    quando ristimare ad ogni passo sarebbe troppo costoso) e si produce la
    previsione one-step-ahead per la presenza SUCCESSIVA, che viene
    confrontata con il voto reale.

CONFRONTO: MAE per-giocatore di SARIMAX sulle sue presenze di test, contro
il MAE dello STESSO giocatore sulle STESSE presenze ottenuto dal modello
LightGBM previsionale già allenato (work/models/lgbm_voto_previsionale_v1.txt).

Output:
  work/data/armax_evaluation_log.txt (log dettagliato + conclusione)
  work/data/armax_per_player_results.csv (MAE per giocatore, entrambi i modelli)

Uso:
  python3 evaluate_armax.py
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_PATH = DATA_DIR / "feature_dataset_v3.csv"
LGBM_MODEL_PATH = MODELS_DIR / "lgbm_voto_previsionale_v1.txt"
LOG_PATH = DATA_DIR / "armax_evaluation_log.txt"
OUT_CSV_PATH = DATA_DIR / "armax_per_player_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("evaluate_armax")

TRAIN_SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                  "2020-21", "2021-22", "2022-23"]
VAL_SEASONS = ["2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]
STAGIONI_ORDINE = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
STAGIONE_IDX = {s: i for i, s in enumerate(STAGIONI_ORDINE)}

MIN_TRAIN_PRESENZE = 80
MIN_TEST_PRESENZE = 8

EXOG_COLS = [
    "giornata",
    "squadra_forma3_punti_mean", "squadra_forma3_gf_mean", "squadra_forma3_gs_mean",
    "squadra_forma3_xg_for_mean", "squadra_forma3_xg_against_mean",
    "squadra_forma5_punti_mean", "squadra_forma5_gf_mean", "squadra_forma5_gs_mean",
    "squadra_forma5_xg_for_mean", "squadra_forma5_xg_against_mean",
    "avversario_forma3_punti_mean", "avversario_forma3_gf_mean", "avversario_forma3_gs_mean",
    "avversario_forma3_xg_for_mean", "avversario_forma3_xg_against_mean",
    "avversario_forma5_punti_mean", "avversario_forma5_gf_mean", "avversario_forma5_gs_mean",
    "avversario_forma5_xg_for_mean", "avversario_forma5_xg_against_mean",
    "voto_lag_mean_3", "voto_lag_mean_5",
    "understat_xG_lag_mean_5", "understat_xA_lag_mean_5",
    "understat_shots_lag_mean_5", "understat_time_lag_mean_5",
    "presenze_cumulate",
    "squadra_posizione_pre", "squadra_punti_pre", "squadra_diff_reti_pre",
    "avversario_posizione_pre", "avversario_punti_pre", "avversario_diff_reti_pre",
    "in_casa",
]

ORDER_GRID = [(1, 0, 0), (1, 0, 1), (2, 0, 1), (2, 1, 1)]

# stesse colonne/categoriche di train_model_previsionale.py, per generare le
# predizioni LightGBM sulle stesse identiche righe usate per SARIMAX
ID_COLS = ["match_id", "data", "nome_giocatore", "player_id", "fonte_voto",
           "senza_voto", "voto", "fantavoto"]
CATEGORICAL_COLS = ["ruolo", "squadra_giocatore", "squadra_casa", "squadra_ospite",
                     "squadra_modulo", "avversario_modulo", "squadra_allenatore", "avversario_allenatore"]
NON_FEATURE_EXTRA = ["stagione"]


def carica_dataset():
    log.info("Caricamento %s ...", FEATURE_PATH)
    df = pd.read_csv(FEATURE_PATH, low_memory=False)
    df = df[df["fonte_voto"] == "redazione"].copy()
    df = df[df["senza_voto"].astype(str) != "True"].copy()
    df = df.dropna(subset=["voto"]).copy()
    df["voto"] = df["voto"].astype(float)
    df["in_casa"] = (df["squadra_giocatore"] == df["squadra_casa"]).astype(float)
    df["stagione_idx"] = df["stagione"].map(STAGIONE_IDX)
    df = df.dropna(subset=["stagione_idx"]).copy()
    log.info("Righe valide totali: %d", len(df))
    return df


def seleziona_giocatori(df):
    train_cnt = df[df["stagione"].isin(TRAIN_SEASONS)].groupby("player_id").size()
    test_cnt = df[df["stagione"].isin(TEST_SEASONS)].groupby("player_id").size()
    both = pd.concat([train_cnt.rename("train"), test_cnt.rename("test")], axis=1).fillna(0)
    qualificati = both[(both["train"] >= MIN_TRAIN_PRESENZE) & (both["test"] >= MIN_TEST_PRESENZE)]
    log.info("Giocatori qualificati per l'esperimento (train>=%d, test>=%d): %d",
              MIN_TRAIN_PRESENZE, MIN_TEST_PRESENZE, len(qualificati))
    return list(qualificati.index)


def prepara_exog_globale(df):
    """Imputa i NaN degli exog con la mediana globale calcolata SOLO sul
    train+val (mai sul test, per evitare qualunque leakage anche nella
    semplice imputazione statistica)."""
    mask_fit = df["stagione"].isin(TRAIN_SEASONS + VAL_SEASONS)
    mediane = df.loc[mask_fit, EXOG_COLS].median()
    log.info("Mediane di imputazione (train+val) calcolate per %d colonne exog", len(mediane))
    return mediane


def predizioni_lgbm(df):
    """Ricrea le feature esattamente come in train_model_previsionale.py e
    produce le predizioni LightGBM per TUTTE le righe, indicizzate per
    match_id+player_id, da confrontare sulle stesse righe di test SARIMAX."""
    feature_cols = [c for c in df.columns if c not in ID_COLS and c not in NON_FEATURE_EXTRA
                     and c not in ("in_casa", "stagione_idx") and c != "giornata"]
    feature_cols = ["giornata"] + feature_cols
    X = df[feature_cols].copy()
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")

    model = lgb.Booster(model_file=str(LGBM_MODEL_PATH))
    pred = model.predict(X)
    return pd.Series(pred, index=df.index)


def stima_ordine(endog_train, exog_train):
    best_aic = np.inf
    best_order = ORDER_GRID[0]
    for order in ORDER_GRID:
        try:
            mod = SARIMAX(endog_train, exog=exog_train, order=order,
                           enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=100)
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = order
        except Exception as e:
            log.debug("Ordine %s fallito: %s", order, e)
            continue
    return best_order, best_aic


def valuta_giocatore(df_player, mediane):
    """Esegue walk-forward SARIMAX per un giocatore, ritorna
    (mae_armax, mae_lgbm, n_test) o None se il fit iniziale fallisce."""
    df_player = df_player.sort_values("stagione_idx").reset_index(drop=True)
    exog_all = df_player[EXOG_COLS].fillna(mediane)
    endog_all = df_player["voto"]

    train_mask = df_player["stagione"].isin(TRAIN_SEASONS + VAL_SEASONS)
    test_mask = df_player["stagione"].isin(TEST_SEASONS)

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    if n_train < MIN_TRAIN_PRESENZE or n_test < MIN_TEST_PRESENZE:
        return None

    endog_train = endog_all[train_mask].reset_index(drop=True)
    exog_train = exog_all[train_mask].reset_index(drop=True)

    order, aic = stima_ordine(endog_train, exog_train)

    try:
        mod = SARIMAX(endog_train, exog=exog_train, order=order,
                       enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=100)
    except Exception as e:
        log.warning("Fit iniziale fallito per giocatore, escluso: %s", e)
        return None

    idx_test = df_player.index[test_mask].tolist()
    errori_armax = []

    stato = res
    for i in idx_test:
        exog_riga = exog_all.loc[[i]]
        try:
            fcst = stato.forecast(steps=1, exog=exog_riga)
            pred = float(fcst.iloc[0])
        except Exception as e:
            log.debug("Forecast fallito su una riga, salto: %s", e)
            continue
        reale = float(endog_all.loc[i])
        errori_armax.append(abs(pred - reale))

        # aggiorna lo stato con l'osservazione reale (refit=False: nessuna
        # ri-stima dei parametri, solo aggiornamento del filtro di Kalman -
        # pratica standard per il walk-forward incrementale)
        try:
            stato = stato.append(endog_all.loc[[i]], exog=exog_riga, refit=False)
        except Exception as e:
            log.debug("Append fallito, mantengo stato precedente: %s", e)

    if not errori_armax:
        return None

    mae_armax = float(np.mean(errori_armax))
    return {
        "n_train": n_train, "n_test": len(errori_armax), "order": order, "aic": aic,
        "mae_armax": mae_armax,
    }


def main():
    df = carica_dataset()
    giocatori = seleziona_giocatori(df)
    mediane = prepara_exog_globale(df)

    log.info("Calcolo predizioni LightGBM previsionale su tutto il dataset...")
    pred_lgbm_series = predizioni_lgbm(df)
    df = df.copy()
    df["pred_lgbm"] = pred_lgbm_series

    risultati = []
    for n_i, pid in enumerate(giocatori, start=1):
        df_p = df[df["player_id"] == pid]
        nome = df_p["nome_giocatore"].iloc[0]
        res = valuta_giocatore(df_p, mediane)
        if res is None:
            log.info("[%d/%d] %s (player_id=%s): esperimento SARIMAX non riuscito, escluso", n_i, len(giocatori), nome, pid)
            continue

        test_mask = df_p["stagione"].isin(TEST_SEASONS)
        df_p_test = df_p.sort_values("stagione_idx")[test_mask]
        mae_lgbm = float((df_p_test["pred_lgbm"] - df_p_test["voto"]).abs().mean())

        risultati.append({
            "player_id": pid, "nome_giocatore": nome,
            "n_train": res["n_train"], "n_test": res["n_test"],
            "order_armax": str(res["order"]), "aic_armax": res["aic"],
            "mae_armax": res["mae_armax"], "mae_lgbm": mae_lgbm,
            "armax_migliore": res["mae_armax"] < mae_lgbm,
        })
        log.info("[%d/%d] %s: MAE ARMAX=%.4f, MAE LGBM=%.4f, ARMAX migliore=%s (n_test=%d, order=%s)",
                  n_i, len(giocatori), nome, res["mae_armax"], mae_lgbm,
                  res["mae_armax"] < mae_lgbm, res["n_test"], res["order"])

    if not risultati:
        log.error("Nessun giocatore valutato con successo. Esperimento inconcludente.")
        return

    res_df = pd.DataFrame(risultati)
    res_df.to_csv(OUT_CSV_PATH, index=False)
    log.info("Risultati per-giocatore scritti in %s", OUT_CSV_PATH)

    n_tot = len(res_df)
    n_armax_migliore = int(res_df["armax_migliore"].sum())
    mae_armax_medio = res_df["mae_armax"].mean()
    mae_lgbm_medio = res_df["mae_lgbm"].mean()
    mae_armax_pesato = np.average(res_df["mae_armax"], weights=res_df["n_test"])
    mae_lgbm_pesato = np.average(res_df["mae_lgbm"], weights=res_df["n_test"])

    log.info("=== RISULTATO FINALE ESPERIMENTO ARMAX ===")
    log.info("Giocatori valutati con successo: %d / %d selezionati", n_tot, len(giocatori))
    log.info("MAE medio (semplice, non pesato) ARMAX: %.4f", mae_armax_medio)
    log.info("MAE medio (semplice, non pesato) LightGBM previsionale: %.4f", mae_lgbm_medio)
    log.info("MAE medio (pesato per n_test) ARMAX: %.4f", mae_armax_pesato)
    log.info("MAE medio (pesato per n_test) LightGBM previsionale: %.4f", mae_lgbm_pesato)
    log.info("Giocatori per cui ARMAX ha MAE inferiore a LightGBM: %d/%d (%.1f%%)",
              n_armax_migliore, n_tot, 100 * n_armax_migliore / n_tot)

    if mae_armax_pesato < mae_lgbm_pesato and n_armax_migliore > n_tot * 0.55:
        conclusione = ("ARMAX risulta competitivo/superiore al gradient boosting su questo "
                        "sottoinsieme di giocatori ad alta numerosita' storica: potrebbe valere "
                        "la pena usarlo come modello ALTERNATIVO o in ENSEMBLE per questa fascia "
                        "di giocatori (quelli con storia lunga e regolare).")
    elif abs(mae_armax_pesato - mae_lgbm_pesato) < 0.02:
        conclusione = ("ARMAX e LightGBM sono sostanzialmente EQUIVALENTI su questo sottoinsieme: "
                        "non emerge un vantaggio chiaro nell'adottare ARMAX, che aggiungerebbe "
                        "complessita' (un modello per-giocatore da mantenere) senza un beneficio "
                        "di accuratezza sufficiente a giustificarla.")
    else:
        conclusione = ("ARMAX NON e' competitivo rispetto al gradient boosting nemmeno sul "
                        "sottoinsieme di giocatori piu' favorevole (storia lunga e regolare): "
                        "il modello cross-giocatore (LightGBM), che sfrutta informazione condivisa "
                        "tra tutti i giocatori/partite, generalizza meglio della stima "
                        "per-singolo-giocatore anche quando la storia individuale e' relativamente lunga. "
                        "Si raccomanda di NON adottare ARMAX per questo problema.")

    log.info("CONCLUSIONE: %s", conclusione)


if __name__ == "__main__":
    main()
