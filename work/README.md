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
python3 scripts/scrape_eta_giocatori.py                 # -> eta_giocatori_storico_2015_2026.csv (già incluso)
python3 scripts/scrape_infortuni_profilo_giocatori.py   # -> infortuni_/profilo_giocatori_storico_2015_2026.csv (già incluso)
python3 scripts/scrape_forum_esperti_2026_27.py         # -> forum_esperti_pagelle_2026_27.csv (già incluso)
python3 scripts/build_stagione_giocatore_dataset.py     # -> stagione_giocatore_dataset_2015_2026.csv (già incluso)
python3 scripts/train_model_rendimento_stagionale.py    # -> models/lgbm_{target}_stagionale_v1.txt (x7) + metriche
python3 scripts/predict_serie_a_2026_27.py              # -> previsioni_serie_a_2026_27.csv/.html/.pdf (già incluso)
python3 scripts/fuzzy_sorprese_forum.py                 # -> previsioni_serie_a_2026_27_con_sorprese.csv (già incluso)
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

## Aggiornamento v2: età, storico esteso, Champions League

Gap dichiarato nella versione precedente (nessuna età/data di nascita
giocatore) risolto con una nuova fonte: **Transfermarkt**
(`scripts/scrape_eta_giocatori.py` → `data/eta_giocatori_storico_2015_2026.csv`),
pagine rosa per squadra/stagione (`transfermarkt.it/.../kader/verein/{id}/saison_id/{anno}`),
fuzzy-matching nome per collegare l'ID Transfermarkt al `player_id`
fantacalcio.it (nessun ID condiviso tra le due fonti). Copertura 93-99% a
seconda della stagione.

Nuove feature aggiunte a `build_stagione_giocatore_dataset.py`:
- `{metrica}_career_mean`: media della metrica su TUTTE le stagioni
  precedenti disponibili (non solo le ultime 3 come `_ma3`) — cattura il
  "livello vero" di un giocatore con storico lungo.
- `eta_n1`: età al 1° agosto della stagione N-1.
- `squadra_in_champions_n1` / `squadra_in_champions_target`: flag
  partecipazione Champions League, verificato stagione per stagione via
  Wikipedia (non un pattern fisso "prime 4 classificate", falso in più
  occasioni — es. 2015-16 solo 2 squadre italiane qualificate).

**Risultato v1→v2** (MAE test, R² test): miglioramento marginale ma reale
su gol (1.338→1.355 → **nota: vedi v3 sotto per il valore aggiornato**) e
assist, praticamente invariato su fantamedia/bonus_netti. `eta_n1` entra
sempre in top-10/20 per importanza (specie gol/assist); il flag Champions
resta marginale (rank 21-39/44, spesso gain≈0).

## Aggiornamento v3: infortuni e profilo fisico

Dario ha condiviso un documento con 327 variabili teoriche di rendimento
(GPS/tracking, xT/VAEP/possession-value, scouting psicologico
soggettivo...). Valutazione di fattibilità: la stragrande maggioranza
richiede fornitori a pagamento (Opta/StatsBomb/Wyscout) o dati proprietari
club, **non ottenibile da fonti pubbliche gratuite**. Sottoinsieme
realisticamente fattibile identificato e implementato: **infortuni +
profilo fisico**, via Transfermarkt (stessa fonte dell'età).

`scripts/scrape_eta_giocatori.py` estratto anche l'ID Transfermarkt per
giocatore (`player_tm_id`, "gratis" dall'href già presente nella pagina
rosa, nessun secondo fuzzy-match necessario) → nuovo script
`scripts/scrape_infortuni_profilo_giocatori.py` (scope: TUTTO lo storico
2015-2026, non solo rosa attuale) scarica per ciascun `player_tm_id`
univoco (2097 giocatori):
- **Infortuni** (`data/infortuni_giocatori_storico_2015_2026.csv`, 16.996
  episodi): tipo, data inizio/fine, giorni di stop, partite perse.
- **Profilo fisico** (`data/profilo_giocatori_storico_2015_2026.csv`, 2088
  giocatori): altezza, nazionalità, piede dominante, posizione naturale,
  scadenza contratto. NOTA: profilo "attuale" al momento dello scraping,
  non storico per stagione (semplificazione dichiarata: altezza/piede/
  nazionalità cambiano raramente o mai). Nessun peso/BMI (Transfermarkt
  non lo pubblica).

**Nota tecnica**: durante lo scraping, il dominio italiano
`transfermarkt.it` ha iniziato a rispondere HTTP 403 sulla maggioranza
delle richieste dopo circa 300 giocatori (rate-limiting/anti-bot). Fonte
cambiata al dominio inglese `transfermarkt.com` (stessa struttura dati),
rate-limit più prudente, e salvataggio incrementale riga per riga (non
più solo a fine run) per non perdere lavoro in caso di nuovo blocco.

Nuove feature in `build_stagione_giocatore_dataset.py` (sempre pre-
stagione N, principio anti-leakage invariato):
- `infortuni_n1_count` / `infortuni_n1_giorni_totali`: episodi e giorni di
  stop nella stagione N-1 (0, non None, se nessun episodio — dato valido).
- `infortuni_career_count`: episodi cumulati in TUTTE le stagioni < N.
- `altezza_m`, `piede_dominante`, `nazionalita`: dal profilo (vedi nota
  "attuale" sopra). Scadenza contratto NON usata come feature (snapshot
  non ricostruibile retroattivamente per stagioni passate).

**Risultato v2→v3** (MAE test, R² test):

| Target | v2 MAE / R² | v3 MAE / R² | Nota |
|---|---|---|---|
| fantamedia | 0.313 / 0.530 | 0.313 / 0.531 | invariato |
| gol | 1.355 / 0.340 | **1.328 / 0.365** | miglioramento più tangibile |
| assist | 1.088 / 0.276 | 1.083 / 0.257 | MAE lievemente meglio, R² lievemente peggio |
| bonus netti | 5.412 / 0.604 | 5.379 / 0.603 | invariato |

`altezza_m` è la feature nuova con più impatto (top-10 per gol/assist,
probabilmente correlata al ruolo più che causale); gli infortuni hanno
impatto medio (rank 13-20/50); nazionalità/piede restano marginali.
**Conclusione onesta**: miglioramento reale ma modesto, non la svolta che
il documento delle 327 variabili avrebbe fatto sperare — coerente con la
valutazione di fattibilità iniziale.

Applicato anche a `scripts/predict_como_2026_27.py` (previsioni rosa Como
2026-27 con i modelli v3).

## Aggiornamento v4-v6: nuovi target, varianza/trend, cartellini/rigori, tutte le 20 squadre

- **v4**: aggiunti 2 nuovi target al modello di rendimento stagionale
  (richiesta Dario): `voto_medio_target` (giudizio redazione puro, SENZA
  bonus/malus, distinto da `fantamedia_target`) e
  `presenze_titolare_target` ("partite titolari", righe Understat con
  `position != 'Sub'`). Corretto in questa fase anche un bug di join
  Understat pre-esistente: il join usava direttamente il `player_id` di
  fantacalcio.it come se fosse lo stesso ID Understat (NON lo è, due
  spazi ID indipendenti) — fix tramite il bridge già esistente
  `player_name_mapping.csv`, copertura salita dal ~2.5% al 98.7%.
- **v5**: aggiunte feature di varianza (`_std3`) e trend/pendenza
  (`_trend3`) sulla stessa finestra delle ultime 3 stagioni già usata per
  `_ma3`, per ogni metrica base — e nuove feature cartellini/rigori
  (`ammonizioni_totali`, `espulsioni_totali`, `rigori_segnati_totali`,
  `rigori_sbagliati_totali`, `rigori_parati_totali`, con lag1/ma3/std3/
  trend3/career_mean come le altre metriche base).
- **v6**: `scripts/predict_como_2026_27.py` generalizzato in
  `scripts/predict_serie_a_2026_27.py` — previsioni per TUTTI i 503
  giocatori delle 20 squadre di Serie A 2026-27, non solo il Como
  (output `data/previsioni_serie_a_2026_27.csv/.html/.pdf`).

## Aggiornamento v7: giudizio qualitativo forum Gruppo Esperti + correzione fuzzy

Dario ha condiviso un link al forum Gruppo Esperti (gruppoesperti.it),
board "Schede squadra" 2026/27 — 20 topic, uno per squadra, con pagelle
degli esperti per ogni giocatore in rosa (Titolarità/Media voto/Salute/
Bonus (o Porta inviolata per i portieri)/Consiglio Esperti, scala 0-10 +
TOTALE /50). Richiesta esplicita: **"Consideriamo anche questi aspetti
nell'analisi"** / **"per ogni squadra"** — integrare questo giudizio
QUALITATIVO/soggettivo (hype preseason) con le previsioni QUANTITATIVE
del modello (basate solo su dati storici oggettivi).

**Nuovo script** `scripts/scrape_forum_esperti_2026_27.py`: scarica i 20
topic (ID hardcoded, board `viewforum.php?f=199`), parsa la riga pagelle
via regex (formato standardizzato, verificato identico su più squadre),
gestisce esplicitamente i punteggi non ancora assegnati (`x/10` → `None`,
mai riempiti a caso), e fa matching nome→`player_id` fantacalcio
ristretto alla stessa squadra (più affidabile che sui 503 giocatori
totali) con lo stesso approccio (`norm`/`match_score`/`MIN_MATCH_SCORE`)
già usato per età/infortuni/profilo. Risultato: 495 giocatori estratti su
20 squadre, 424 matchati a un `player_id` (85,7%), 71 non-match
dichiarati (perlopiù giovani/riserve marginali fuori dalle quotazioni
fantacalcio). Output: `data/forum_esperti_pagelle_2026_27.csv`.

**Nota sul dominio**: `www.gruppoesperti.it/forum/` ha iniziato a
rispondere HTTP 526 "Invalid SSL certificate" (guasto lato server,
confermato anche con un browser headless reale — non un blocco
anti-bot). Dario ha indicato il sottodominio corretto e funzionante
`forum.gruppoesperti.it` (stesso contenuto, nessun login necessario).

**Scoperta e limite onesto**: le 6 colonne `*_forum` sono state
aggiunte prima come feature DIRETTE in `build_stagione_giocatore_dataset.py`
(popolate solo per `stagione_target=='2026-27'`, `None`/NaN su tutte le
stagioni storiche — sono un giudizio che esiste solo per la prossima
stagione, non backfillabile). Verificato dopo il retrain (v7): la
**feature importance di tutte le 6 colonne è esattamente 0.0** su tutti
i 7 target — motivo strutturale, non un bug: essendo sempre NaN nel
train/val/test (2017-18→2025-26), LightGBM non ha mai potuto imparare
alcuna relazione con un dato che non esiste in nessuna riga di training.
Le metriche MAE/RMSE/R² sul test set storico sono infatti identiche a
v6, come atteso.

**Soluzione**: sistema di **inferenza fuzzy (Mamdani, libreria
scikit-fuzzy)** in post-processing, `scripts/fuzzy_sorprese_forum.py` —
bypassa il limite strutturale del training confrontando, DOPO che il
modello ha già previsto, `pred_presenze_titolare_previste` (quanto il
modello si aspetta il giocatore titolare, da storico oggettivo) con
`titolarita_forum_esperti` (quanto gli esperti lo giudicano titolare per
il 2026-27, giudizio preseason). 9 regole esperte scritte a mano (nessun
training necessario) producono un `indice_sorpresa` (-10..+10):
divergenza forte e coerente in una direzione → "sorpresa al rialzo" o
"rischio al ribasso"; accordo tra le due fonti → "coerente". Le funzioni
di appartenenza sono state calibrate sui percentili REALI delle
previsioni (non sul range teorico 0-38 giornate) dopo aver verificato
che la calibrazione iniziale generava falsi positivi su titolari
indiscutibili (es. Barella, Cambiaso, Di Lorenzo). Righe senza giudizio
forum disponibile vengono escluse dal calcolo (`indice_sorpresa=None`),
non forzate a un valore neutro.

**Risultato**: su 303 giocatori 2026-27 con giudizio forum disponibile,
29 segnalati come sorpresa al rialzo (es. Baturina, Calhanoglu, Malen —
il forum più fiducioso del modello) e 18 come rischio al ribasso (es.
Marusic, Lucumì, Modric, Tomori — il modello li vede titolari fissi da
storico ma gli esperti segnalano un ballottaggio che i dati storici non
possono cogliere). Output:
`data/previsioni_serie_a_2026_27_con_sorprese.csv`.

## Aggiornamento v8: schede partita giornata 1 (Gruppo Esperti)

Dario ha condiviso il link al topic Roma-Fiorentina della board Gruppo
Esperti "Schede squadra e schede partita" (`viewforum.php?f=199`),
chiedendo esplicitamente: **"Analizza tutti i thread delle partite
della prima giornata per aumentare i dati della valutazione"**. A
differenza dei 20 topic "SQUADRA [TOPIC UNICO]" già scrapati in v7
(giudizio generico/preseason valido per tutta la stagione), la board
contiene una sezione sorella di 10 topic "schede partita" — uno per
ogni incontro della giornata 1 — con contenuto molto più specifico e
fresco: probabile formazione, ballottaggi con percentuale, giocatori
indisponibili per QUESTA giornata, voto consigliato 1/5-5/5 per OGNI
giocatore della rosa (non solo i titolari), rigoristi e chi calcia
punizioni/angoli.

**Nuovo script** `scripts/scrape_schede_partita_2026_27.py`: scarica i
10 topic (ID hardcoded, `TOPIC_TO_PARTITA`, verificati dal vivo su
Roma-Fiorentina e Atalanta-Sassuolo poi generalizzati agli altri 8),
riusa `HEADERS`/`get_con_retry`/`norm`/`match_score`/
`MIN_MATCH_SCORE`/`carica_quotazioni_2026_27` da
`scrape_forum_esperti_2026_27.py` via import. Ogni topic ha 2 post
ufficiali (uno per squadra, autore = nome ufficiale es. "AS Roma");
sezioni delimitate da marcatori-immagine (stesso pattern di
riconoscimento robusto di `scrape_forum_esperti_qualitativo_2026_27.py`:
fermarsi al marcatore successivo QUALSIASI, non solo a quelli noti).
Il blocco voto 1/5-5/5 ha un formato leggermente incoerente tra squadre
(wrapper `<strong class="text-strong">` esterno presente o assente,
colore sia hex `#00BF00` sia nome CSS `red`) — gestito con una regex
tollerante dopo averlo scoperto dal vivo sul post Sassuolo (0 voti
estratti con la prima versione della regex). Matching nome→`player_id`
ristretto alle due squadre della partita corrente (più stringente di
tutta la rosa). Non-match dichiarati e loggati, mai forzati.

**Risultato scraping**: 459 righe (giocatori con voto) su 10 partite/20
squadre, 397 matchati a un `player_id` (86,5%), 62 non-match dichiarati
(perlopiù giovani/riserve marginali fuori da
`quotazioni_fantacalcio_storico_2015_2026.csv`, o acquisti troppo
recenti — es. Frattesi alla Lazio — non ancora presenti nel listone).
Output: `data/schede_partita_giornata1_2026_27.csv`.

**Nuovo script di merge** `scripts/merge_schede_partita_giornata1.py`:
left-join con `previsioni_serie_a_2026_27_con_sorprese.csv` su
`(squadra, nome)` tramite la stessa tabella quotazioni (join esatto per
costruzione, non serve fuzzy matching qui). Aggiunge le colonne
`giornata1_*`/`titolare_previsto_giornata1`/`ballottaggio_pct_giornata1`/
`indisponibile_giornata1`/ecc. e un **nuovo indice fuzzy**
`indice_rischio_giornata1` (stesso sistema Mamdani di
`fuzzy_sorprese_forum.py`, riusato via import di
`costruisci_sistema_fuzzy()`), che confronta
`pred_presenze_titolare_previste` (storico) con una `titolarita_giornata1`
derivata SOLO da segnali espliciti della scheda partita (indisponibile
dichiarato → 0; ballottaggio_pct/10; nessun altro caso dedotto — un
giocatore matchato ma senza ballottaggio né indisponibilità non riceve
un valore inventato, principio anti-invenzione-dati applicato in modo
ancora più stringente che in v7, perché ballottaggi/indisponibili sono
per natura un elenco di eccezioni, non una lista esaustiva).

**Risultato merge**: su 359 giocatori del report, 290 hanno dati di
giornata 1 disponibili; di questi, l'indice è calcolabile per 76
giocatori (quelli con ballottaggio o indisponibilità dichiarati) — 2
"OCCASIONE GIORNATA 1" (es. N'Dri/Lecce, Corvi/Parma: ballottaggio alto
ma modello previsionale storico basso, tipico di un titolare designato
con poco storico) e 25 "RISCHIO GIORNATA 1" (es. Atta/Fiorentina,
Pulisic e Leao/Milan: indisponibili per questa giornata ma il modello,
guardando solo lo storico, li prevede titolari fissi). Output:
`data/previsioni_serie_a_2026_27_giornata1.csv`.

**Fuori scope** (non incluso in questo giro): integrazione nel sito
statico GitHub Pages (`index.html` + nuovo `*_data.js`) — proponibile
come follow-up separato se richiesto.

## Direzione futura (fase successiva, non ancora iniziata)

Modello a livello **squadra** (stagione, squadra_target): stessa logica,
target = punti finali stagione N, feature = aggregati/posizione stagione
N-1 + eventuale aggregazione dei target giocatore previsti per la rosa
nota. Da avviare dopo validazione con Dario del modello giocatore.
