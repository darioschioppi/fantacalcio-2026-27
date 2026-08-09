# Dati e script di lavoro — modello ML valutazione fantacalcio

Questa cartella contiene lo storico dati e gli script Python usati per costruire
un modello ML che predice il voto fantacalcio di un giocatore, integrando fonti
esterne (fantacalcio.it, Lega Serie A / Deltatre, Understat).

## Struttura

- `scripts/` — script Python di scraping e feature engineering (ordine di esecuzione tipico sotto)
- `data/` — dataset grezzi/intermedi aggregati (11 stagioni, 2015-16 → 2025-26) + log di scraping/validazione
- `models/` — modello LightGBM allenato (`lgbm_voto_v1.txt`)

## Dataset NON incluso in questo repo

I file `feature_dataset_v0/v1/v2/v3.csv` (join finali usati per il training,
68-112MB) NON sono versionati qui perché superano/si avvicinano ai limiti
dimensione file di GitHub e sono interamente RIPRODUCIBILI dai dati grezzi
già presenti eseguendo in ordine:

```
python3 scripts/build_feature_dataset.py         # -> feature_dataset_v1.csv
python3 scripts/build_classifica_dinamica.py      # -> classifica_dinamica_storico_2015_2026.csv (già incluso)
python3 scripts/build_feature_dataset_v2.py       # -> feature_dataset_v2.csv
python3 scripts/train_model.py                    # -> models/lgbm_voto_v1.txt + metriche

python3 scripts/build_squadra_form_dinamica.py    # -> squadra_form_dinamica_storico_2015_2026.csv (già incluso)
python3 scripts/build_player_lag_features.py      # -> player_lag_features_storico_2015_2026.csv (già incluso)
python3 scripts/build_feature_dataset_v3.py       # -> feature_dataset_v3.csv (solo feature pre-partita)
python3 scripts/train_model_previsionale.py       # -> models/lgbm_voto_previsionale_v1.txt + metriche
python3 scripts/evaluate_armax.py                 # -> armax_evaluation_log.txt + armax_per_player_results.csv (già incluso)
```

## Fonti dati

- **Voti/pagelle**: `https://www.fantacalcio.it/voti-fantacalcio-serie-a/{stagione}/{giornata}` (pagina pubblica, no auth)
- **Statistiche squadra + formazioni/allenatori**: API Deltatre di Lega Serie A (`seriea-api.prd.sdp.deltatre.digital`, no auth)
- **Statistiche individuali per partita (xG/xA/tiri)**: Understat.com (nota: `robots.txt: Disallow: /` — scraping a rate limitato per uso interno, non ripubblicazione)

## Stato modello v1 ("explanatory", target = voto singola partita)

Target: `voto` (giudizio redazione). Split temporale (train 2015-16→2022-23,
val 2023-24, test 2024-25/2025-26). MAE test 0.31, RMSE 0.41, R² 0.55.

**Nota importante**: questo modello è "explanatory" (ricostruisce il voto
dalle statistiche della partita già giocata: gol, xG, minuti, cartellini),
NON genuinamente previsionale. Utile come strumento di analisi
retrospettiva, non come pronostico pre-partita.

## Esperimento: modello PREVISIONALE (target = voto singola partita, solo feature pre-fischio d'inizio)

Per verificare se il voto della singola partita fosse predicibile PRIMA che
si giocasse, si è costruito un secondo dataset (`feature_dataset_v3.csv`,
via `build_squadra_form_dinamica.py` + `build_player_lag_features.py` +
`build_feature_dataset_v3.py`) con SOLO feature disponibili prima del
fischio d'inizio: forma rolling squadra/avversario (ultime 3-5 partite),
storia recente del giocatore (lag di voto/xG/xA/minuti), classifica
pre-partita, modulo/allenatore. Verificata automaticamente l'assenza di
ogni colonna di risultato/bonus della partita corrente.

**Risultato (`train_model_previsionale.py`, modello `lgbm_voto_previsionale_v1.txt`)**:
MAE test 0.4399, RMSE 0.5986, R² 0.0111 — praticamente identico alla
baseline "predici sempre la media" (MAE 0.4422, R² ≈ 0). **Conclusione: il
voto della singola partita non è predicibile in modo utile con il contesto
pre-partita disponibile** — il segnale dominante sono gli eventi della
partita stessa (episodi, minutaggio, decisioni arbitrali), non il contesto
conosciuto in anticipo.

## Esperimento: ARMAX/SARIMAX per-giocatore

Valutata la fattibilità di ARMAX (`SARIMAX` di `statsmodels`) come modello
alternativo, su un sottoinsieme scoping di 114 giocatori con storia
sufficiente (≥80 presenze train, ≥8 presenze test — la mediana storica è
28 presenze/giocatore su tutta la popolazione, insufficiente per stimare un
ARIMA per-giocatore in modo stabile). Validazione walk-forward one-step-ahead,
confronto diretto contro il modello previsionale LightGBM sulle stesse
partite (`evaluate_armax.py`, risultati in `armax_per_player_results.csv`).

**Risultato**: MAE medio ARMAX 0.674 vs MAE medio LightGBM 0.450; ARMAX
migliore in soli 9/114 casi (7.9%). **Conclusione: ARMAX scartato**, non
competitivo nemmeno sul sottoinsieme più favorevole.

## Direzione futura (in corso): rendimento stagionale aggregato, non voto singola partita

A valle di questi due esperimenti, l'obiettivo è stato ridefinito: **non**
predire il voto di una singola partita (dimostrato quasi impredicibile),
ma **il rendimento aggregato di un giocatore/squadra sull'arco della
stagione** (es. fantamedia finale, presenze, gol/assist totali attesi) —
un problema più adatto a supportare decisioni d'asta (chi comprare e a
quale prezzo), aggregando su molte partite invece di indovinare il singolo
episodio imprevedibile.
