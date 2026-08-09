# Dati e script di lavoro — modello ML valutazione fantacalcio

Questa cartella contiene lo storico dati e gli script Python usati per costruire
un modello ML che predice il voto fantacalcio di un giocatore, integrando fonti
esterne (fantacalcio.it, Lega Serie A / Deltatre, Understat).

## Struttura

- `scripts/` — script Python di scraping e feature engineering (ordine di esecuzione tipico sotto)
- `data/` — dataset grezzi/intermedi aggregati (11-12 stagioni, 2015-16 → 2026-27) + log di scraping/validazione
- `models/` — modelli LightGBM allenati (voto singola partita + rendimento stagionale per target)

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

python3 scripts/scrape_quotazioni.py                    # -> quotazioni_fantacalcio_storico_2015_2026.csv (già incluso)
python3 scripts/build_stagione_giocatore_dataset.py     # -> stagione_giocatore_dataset_2015_2026.csv (già incluso)
python3 scripts/train_model_rendimento_stagionale.py    # -> models/lgbm_{target}_stagionale_v1.txt (x4) + metriche
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

## Rendimento stagionale aggregato (per decisioni d'asta)

A valle dei due esperimenti precedenti, l'obiettivo è stato ridefinito: **non**
predire il voto di una singola partita (dimostrato quasi impredicibile),
ma **il rendimento aggregato di un giocatore sull'arco della stagione** —
un problema più adatto a supportare decisioni d'asta (chi comprare e a
quale prezzo), aggregando su molte partite invece di indovinare il singolo
episodio imprevedibile.

Nuova unità statistica: una riga = (player_id, stagione_target). Problema
PRE-STAGIONE: si prevede la stagione N usando solo dati fino a fine
stagione N-1 (+ quotazione iniziale della stagione N stessa, pubblicata
prima del campionato quindi non leakage).

**4 target separati** (richiesti esplicitamente da Dario, un modello
LightGBM per ciascuno):
- `fantamedia_target` — media fantavoto redazione sulla stagione
- `gol_target` — somma gol fatti sulla stagione
- `assist_target` — somma assist sulla stagione
- `bonus_netti_target` — somma di (fantavoto - voto) sulla stagione (bonus/malus
  netti già calcolati dalla redazione riga per riga: verificato essere sempre
  un multiplo esatto di 0.5 su tutte le presenze valide)

**Fonti dati integrate**:
- `voti_storici_2015_2026.csv` aggregato per (player_id, stagione)
- Understat aggregato (xG/xA/shots/minuti totali stagione)
- `classifica_dinamica_storico_2015_2026.csv` ridotto a fine-stagione (punti/posizione finale)
- **Quotazioni ufficiali Fantacalcio.it** (`scrape_quotazioni.py` →
  `quotazioni_fantacalcio_storico_2015_2026.csv`, scraping read-only delle
  pagine pubbliche `fantacalcio.it/quotazioni-fantacalcio/{stagione}`,
  nessun login richiesto, join diretto per player_id): quotazione
  iniziale/attuale (in crediti, coprono tutte le 11 stagioni storiche) e
  FVM (Fanta Valore di Mercato, "previsione di spesa" del sito,
  disponibile solo dal 2022-23 in avanti — 4 stagioni su 11). FVM è
  pubblicato in scala di riferimento "/1000"; riscalato qui `/2` per il
  budget da 500 crediti della lega di Dario e usato come feature
  aggiuntiva (`fvm_n1`, solo stagione N-1 per restare pre-stagione), senza
  sostituire quotazione iniziale/attuale che restano la feature principale
  per copertura storica completa.

**Feature per riga** (SOLO da stagioni ≤ N-1, principio anti-leakage
verificato automaticamente con `SystemExit(1)` se violato): aggregati
lag1 e media mobile 3 stagioni delle metriche base, ruolo, cambio squadra
(+ contesto forza squadra vecchia/nuova), presenze cumulate carriera,
quotazione iniziale/attuale N-1, FVM N-1, quotazione iniziale della
stagione target (unica eccezione dichiarata: nota prima che la stagione
cominci).

**Risultato (`train_model_rendimento_stagionale.py`)** — split temporale
train 2017-18→2022-23, val 2023-24, test 2024-25/2025-26 (n=692), 3
baseline di confronto per ciascun target (media train, "ripeti anno
precedente", regressione lineare sulla sola quotazione ufficiale):

| Target | MAE test | R² test | R² baseline quotazione | R² baseline "anno scorso" |
|---|---|---|---|---|
| fantamedia | 0.31 | 0.53 | 0.24 | 0.20 |
| gol | 1.34 | 0.33 | 0.24 | 0.05 |
| assist | 1.08 | 0.26 | 0.16 | -0.07 |
| bonus netti | 5.41 | 0.60 | 0.13 | 0.44 |

**Conclusione: tutti e 4 i modelli battono nettamente tutte le baseline**
sul test set — a differenza della fase precedente (voto singola partita),
l'aggregazione stagionale rende il rendimento del giocatore effettivamente
predicibile in modo utile, confermando l'ipotesi di Dario sul framing del
problema.

Output: `models/lgbm_{fantamedia,gol,assist,bonus_netti}_stagionale_v1.txt`,
`data/model_train_stagionale_log.txt`.

## Direzione futura (fase successiva, non ancora iniziata)

Modello a livello **squadra** (stagione, squadra_target): stessa logica,
target = punti finali stagione N, feature = aggregati/posizione stagione
N-1 + eventuale aggregazione dei target giocatore previsti per la rosa
nota. Da avviare dopo validazione con Dario del modello giocatore. Gap
dichiarato non ancora risolto: nessuna età/data di nascita giocatore in
nessuna fonte usata finora.
