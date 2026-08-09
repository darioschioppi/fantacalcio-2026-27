# Dati e script di lavoro — modello ML valutazione fantacalcio

Questa cartella contiene lo storico dati e gli script Python usati per costruire
un modello ML che predice il voto fantacalcio di un giocatore, integrando fonti
esterne (fantacalcio.it, Lega Serie A / Deltatre, Understat).

## Struttura

- `scripts/` — script Python di scraping e feature engineering (ordine di esecuzione tipico sotto)
- `data/` — dataset grezzi/intermedi aggregati (11 stagioni, 2015-16 → 2025-26) + log di scraping/validazione
- `models/` — modello LightGBM allenato (`lgbm_voto_v1.txt`)

## Dataset NON incluso in questo repo

I file `feature_dataset_v0/v1/v2.csv` (join finali usati per il training,
68-108MB) NON sono versionati qui perché superano/si avvicinano ai limiti
dimensione file di GitHub e sono interamente RIPRODUCIBILI dai dati grezzi
già presenti eseguendo in ordine:

```
python3 scripts/build_feature_dataset.py        # -> feature_dataset_v1.csv
python3 scripts/build_classifica_dinamica.py     # -> classifica_dinamica_storico_2015_2026.csv (già incluso)
python3 scripts/build_feature_dataset_v2.py      # -> feature_dataset_v2.csv
python3 scripts/train_model.py                   # -> models/lgbm_voto_v1.txt + metriche
```

## Fonti dati

- **Voti/pagelle**: `https://www.fantacalcio.it/voti-fantacalcio-serie-a/{stagione}/{giornata}` (pagina pubblica, no auth)
- **Statistiche squadra + formazioni/allenatori**: API Deltatre di Lega Serie A (`seriea-api.prd.sdp.deltatre.digital`, no auth)
- **Statistiche individuali per partita (xG/xA/tiri)**: Understat.com (nota: `robots.txt: Disallow: /` — scraping a rate limitato per uso interno, non ripubblicazione)

## Stato modello (v1)

Target: `voto` (giudizio redazione). Split temporale (train 2015-16→2022-23,
val 2023-24, test 2024-25/2025-26). MAE test 0.31, RMSE 0.41, R² 0.55.

**Nota importante**: il modello attuale è principalmente "explanatory"
(ricostruisce il voto dalle statistiche della partita già giocata) più che
genuinamente "previsionale" (predire il voto PRIMA della partita). Prossimo
step: feature con statistiche "lag" (medie mobili partite precedenti) invece
delle statistiche della partita corrente.
