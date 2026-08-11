#!/usr/bin/env python3
"""
Costruisce il dataset a livello STAGIONE-GIOCATORE per il nuovo obiettivo
richiesto da Dario: non predire il voto di una singola partita (dimostrato
quasi impossibile nelle fasi precedenti: modello previsionale v3 MAE~=
baseline, ARMAX non competitivo), ma predire il RENDIMENTO AGGREGATO di un
giocatore sull'arco di UNA STAGIONE INTERA, utile per decidere se comprarlo
all'asta del fantacalcio e a quale prezzo.

UNITA' STATISTICA: una riga = (player_id, stagione_target). Il problema è
PRE-STAGIONE: si vuole prevedere il rendimento della stagione N usando SOLO
informazione nota PRIMA che la stagione N cominci (storico fino a fine
stagione N-1 + eventuale quotazione iniziale della stagione N stessa, che è
pubblicata prima del campionato quindi non è leakage).

TARGET (4, tutti richiesti esplicitamente da Dario: "fantamedia redazione
fantacalcio di ogni singolo giocatore e anche gol, assist e bonus"):
  - fantamedia_target       = media di "fantavoto" sulle presenze valide
                                della stagione N (fantamedia ufficiale)
  - gol_target              = somma di "gol_fatti" sulla stagione N
  - assist_target           = somma di "assist" sulla stagione N
  - bonus_netti_target      = somma di (fantavoto - voto) sulla stagione N.
                                NOTA: fantavoto-voto è il bonus/malus netto
                                GIA' CALCOLATO dalla redazione riga per
                                riga (verificato: sempre multiplo esatto di
                                0.5 su tutte le 114.111 presenze valide) -
                                si usa questa differenza diretta invece di
                                ricostruire pesi bonus/malus a mano, più
                                robusto e non richiede assumere un
                                regolamento specifico non confermato.
  - presenze_target (contesto, NON un target da prevedere con priorità ma
    riportato sempre accanto ai target: una fantamedia alta con poche
    presenze vale meno all'asta di una fantamedia alta con stagione piena)

Definizione di "presenza valida" nella stagione: fonte_voto=="redazione" e
senza_voto!="True" (stessa definizione usata in tutte le fasi precedenti).

FEATURE (tutte calcolate usando SOLO dati di stagioni <= N-1, MAI della
stagione N che si vuole prevedere - stesso principio anti-leakage delle
fasi precedenti, qui a grana stagionale invece che a grana partita):
  - Aggregati del giocatore nella stagione N-1 (le stesse 4 metriche +
    presenze, "_lag1"): componente autoregressiva diretta.
  - Media mobile delle stesse metriche sulle ultime fino a 3 stagioni
    precedenti (N-1, N-2, N-3 se disponibili): cattura un trend più stabile
    del solo ultimo anno (che può essere anomalo per infortuni/big
    minutes).
  - Aggregati Understat (xG/xA/shots totali, minuti totali) della stagione
    N-1, da understat_player_match_stats_storico_2015_2026.csv aggregato
    per (player_id, stagione).
  - ruolo (il più frequente in carriera fino a N-1).
  - squadra_giocatore_N1 (squadra nella stagione N-1) e flag
    cambio_squadra (1 se la squadra nota per la stagione N è diversa da
    quella di N-1, 0 altrimenti; None se la squadra N non è nota - per le
    stagioni storiche già concluse la squadra N è nota dai voti stessi,
    per un uso futuro pre-asta 2026-27 andrebbe fornita esternamente).
  - Contesto squadra: punti finali e posizione finale della squadra del
    giocatore nella stagione N-1 (da classifica_dinamica_storico, ultima
    giornata=38 di quella stagione => è la classifica FINALE, quindi nota
    per costruzione a fine N-1, prima che inizi N).
  - presenze_cumulate_carriera_N1 = numero di stagioni precedenti a N in
    cui il giocatore ha avuto almeno 1 presenza valida (proxy indiretto di
    esperienza/età, in assenza di data di nascita nei dati).
  - `{metrica}_career_mean`: media della metrica su TUTTE le stagioni
    precedenti disponibili del giocatore (non solo le ultime 3 come
    ma3) - richiesto da Dario ("Bisogna trovare il modo da considerare
    più fattori", in seguito al test sulla rosa Como 2026-27 dove ha
    notato che il modello si basava quasi solo sulla stagione precedente).
    Cattura il "livello vero" di un giocatore con storico lungo, a
    differenza di lag1/ma3 che pesano solo gli ultimi 1-3 anni.
  - `eta_n1`: età del giocatore al 1° agosto della stagione N-1 (sempre
    nota per costruzione, calcolata da data di nascita - vedi
    scrape_eta_giocatori.py/Transfermarkt). None se il giocatore non è
    stato matchato alla fonte età (non e' un errore, va dichiarato).
  - `squadra_in_champions_n1` / `squadra_in_champions_target`: flag 1/0,
    la squadra del giocatore giocava/gioca la Champions League in quella
    stagione. Tabella SQUADRE_CHAMPIONS sotto, verificata via ricerca
    web stagione per stagione (fetch diretto pagine "fase a gironi"
    Wikipedia) - NON assunta "sempre le prime 4 classificate" perché
    falso in alcune stagioni (es. 2015-16: solo Juventus e Roma, Inter
    4a classificata non qualificata; 2023-24: 5 italiane per posto extra
    da ranking UEFA). squadra_in_champions_target NON è leakage: le
    qualificazioni Champions sono decise a fine della stagione N-1,
    quindi note prima che la stagione N cominci (stesso principio già
    usato per quotazione_iniziale_target).
  - Quotazione ufficiale Fantacalcio.it (Classic): quotazione_iniziale_N1
    (quotazione a inizio della stagione N-1, quindi doppiamente "safe":
    nota MOLTO prima della stagione N) e, quando disponibile,
    quotazione_iniziale_N (quotazione a inizio della stagione N STESSA -
    NON e' leakage: e' il prezzo d'asta ufficiale pubblicato PRIMA che la
    stagione N cominci, esattamente l'informazione disponibile a chi deve
    decidere se comprare il giocatore all'asta).
  - AGGIORNAMENTO (Dario ha condiviso un PDF con 327 variabili teoriche di
    rendimento; verificato che la maggioranza NON e' ottenibile da fonti
    gratuite, ma infortuni + profilo fisico SI', via Transfermarkt -
    scrape_infortuni_profilo_giocatori.py):
    - `infortuni_n1_count`: numero di episodi di infortunio del giocatore
      con anno di inizio riconducibile alla stagione N-1 (0 se nessuno,
      non None - un giocatore senza infortuni quella stagione e'
      un'informazione valida, non un dato mancante).
    - `infortuni_n1_giorni_totali`: somma dei giorni di stop per quegli
      stessi episodi N-1 (0.0 se nessuno).
    - `infortuni_career_count`: numero totale di episodi in TUTTE le
      stagioni precedenti a N (< N), stesso principio di `_career_mean`
      - cattura la "fragilita' storica" di un giocatore, non solo
      l'ultimo anno.
    - `altezza_m`, `piede_dominante` (categoriale come `ruolo`),
      `nazionalita` (categoriale): dal profilo Transfermarkt, che e'
      "attuale" al momento dello scraping (10/08/2026), non storico per
      stagione - semplificazione dichiarata esplicitamente: altezza/piede/
      nazionalita' non cambiano nel tempo (o quasi mai), quindi usare il
      valore attuale anche per stagioni passate e' un'approssimazione
      accettabile, A DIFFERENZA di un dato come la quotazione che varia
      stagione per stagione e per cui questo NON sarebbe accettabile.
      None se il player_id non e' stato scrappato/matchato - dichiarato,
      non riempito a caso.
    - Scadenza contratto NON e' usata come feature quantitativa nel
      training storico (e' uno snapshot "ad oggi", non ricostruibile
      retroattivamente per le stagioni passate in modo affidabile - stesso
      principio del punto sopra ma qui la variabile CAMBIA nel tempo in
      modo non recuperabile da uno snapshot unico, quindi si esclude
      invece di introdurre rumore/leakage silenzioso).
  - FVM (Fanta Valore di Mercato) della stagione N-1, `fvm_n1` - Dario ha
    confermato: "il FVM è una previsione di spesa fatta da fantacalcio"
    (una stima del sito, non solo un derivato meccanico di QI/QA) e ha
    chiesto di tenerlo COME FEATURE AGGIUNTIVA, non in sostituzione di
    QI/QA (che restano la feature "quotazione" principale perché coprono
    tutte le 11 stagioni storiche, mentre FVM è disponibile solo dal
    2022-23 in avanti - 4 stagioni su 11, verificato: 0 valori non-nulli
    2015-16..2021-22). FVM è pubblicato dal sito in scala "/1000" (budget
    d'asta di riferimento a 1000 crediti), mentre la lega di Dario usa un
    budget totale di 500 crediti: si riscala qui `fvm/2` per essere
    direttamente comparabile a quotazione_iniziale/attuale (già in scala
    "crediti reali" 500). Nota: NON si usa `fvm_target` (stagione target
    stessa) come feature, perché numericamente risulta più vicino alla
    quotazione ATTUALE (fine stagione, aggiornata col rendimento) che alla
    quotazione INIZIALE (MAE fvm/2 vs QA ~5-8 crediti, sistematicamente
    più basso di MAE fvm/2 vs QI sulle 4 stagioni con dato disponibile) -
    quindi riflette informazione di fine-stagione N, non pre-stagione:
    usarlo per la stagione target sarebbe leakage. Resta quindi solo
    `fvm_n1` (stagione N-1, sempre sicura) tra le feature, ed è già
    esplicitamente esclusa `fvm_target` in COLONNE_VIETATE più sotto.

VERIFICA ANTI-LEAKAGE: le colonne di output vengono controllate
automaticamente contro un insieme di pattern vietati (qualunque colonna
che rifletta un aggregato della stagione N stessa, salvo i target
espliciti e quotazione_iniziale_N che e' dichiaratamente pre-stagione) -
interruzione con SystemExit(1) se viene trovata una violazione.

Split temporale (invariato rispetto alle fasi precedenti, spostato da
partita a "stagione target"): train = stagioni target 2017-18..2022-23
(le primissime stagioni 2015-16/2016-17 servono solo a costruire lo
storico iniziale, non generano righe target perché non hanno storico
N-1/N-2/N-3 sufficiente), val = 2023-24, test = 2024-25 e 2025-26.

Output:
  work/data/stagione_giocatore_dataset_2015_2026.csv
  work/data/build_stagione_giocatore_dataset_log.txt

Uso:
  python3 build_stagione_giocatore_dataset.py
"""
import csv
import logging
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
UNDERSTAT_PATH = DATA_DIR / "understat_player_match_stats_storico_2015_2026.csv"
MAPPING_PATH = DATA_DIR / "player_name_mapping.csv"
CLASSIFICA_PATH = DATA_DIR / "classifica_dinamica_storico_2015_2026.csv"
QUOTAZIONI_PATH = DATA_DIR / "quotazioni_fantacalcio_storico_2015_2026.csv"
ETA_PATH = DATA_DIR / "eta_giocatori_storico_2015_2026.csv"
INFORTUNI_PATH = DATA_DIR / "infortuni_giocatori_storico_2015_2026.csv"
PROFILO_PATH = DATA_DIR / "profilo_giocatori_storico_2015_2026.csv"
FORUM_ESPERTI_PATH = DATA_DIR / "forum_esperti_pagelle_2026_27.csv"
OUT_PATH = DATA_DIR / "stagione_giocatore_dataset_2015_2026.csv"
LOG_PATH = DATA_DIR / "build_stagione_giocatore_dataset_log.txt"

# Mappa nomi squadra fantacalcio.it -> nome squadra Understat (team_title),
# stessa mappa di build_feature_dataset.py (le 3 uniche squadre con nome
# diverso tra le due fonti su tutto lo storico 2015-2026).
TEAM_NAME_MAP_UNDERSTAT = {
    "SPAL": "SPAL 2013",
    "Milan": "AC Milan",
    "Parma": "Parma Calcio 1913",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("build_stagione_giocatore_dataset")

STAGIONI_STORICHE = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
                      "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
STAGIONE_IDX = {s: i for i, s in enumerate(STAGIONI_STORICHE)}

# Sigla squadra usata nelle quotazioni -> nome squadra usato in voti_storici/classifica_dinamica.
# Verificato dal vivo: le sigle sono le prime 3 lettere del nome (maiuscole),
# tranne pochi casi ambigui gestiti esplicitamente sotto.
SIGLA_TO_NOME = {
    "ATA": "Atalanta", "BOL": "Bologna", "CAR": "Carpi", "CHI": "Chievo",
    "EMP": "Empoli", "FIO": "Fiorentina", "FRO": "Frosinone", "GEN": "Genoa",
    "INT": "Inter", "JUV": "Juventus", "LAZ": "Lazio", "MIL": "Milan",
    "NAP": "Napoli", "PAL": "Palermo", "ROM": "Roma", "SAM": "Sampdoria",
    "SAS": "Sassuolo", "TOR": "Torino", "UDI": "Udinese", "VER": "Verona",
    "CRO": "Crotone", "BEN": "Benevento", "SPA": "SPAL", "PES": "Pescara",
    "PAR": "Parma", "BRE": "Brescia", "LEC": "Lecce", "CAG": "Cagliari",
    "SPE": "Spezia", "SAL": "Salernitana", "VEN": "Venezia", "COM": "Como",
    "MON": "Monza", "CRE": "Cremonese", "PIS": "Pisa",
}

# colonne dei target/aggregati calcolate per una data stagione (usate sia
# per il target della stagione N sia, con suffisso _lagK, per le feature
# delle stagioni precedenti)
# AGGIORNAMENTO v4 (richiesta Dario: "partite titolari" nella lista
# completa): aggiunta 'presenze_titolare', possibile SOLO ora che il bug di
# join Understat e' stato corretto (vedi carica_name_mapping/
# trova_understat_agg) - conta le righe Understat del giocatore in quella
# stagione con position != 'Sub' (colonna 'position' di Understat: 'Sub'
# flagga in modo affidabile i subentrati, verificato con un sanity-check
# sui minuti medi: mediana 17 min per 'Sub' vs 90 min per le altre
# posizioni). E' un conteggio di "partite da titolare" nel senso di
# "sceso in campo dal 1'", non richiede una soglia minuti arbitraria.
METRICHE_BASE = ["fantamedia", "gol", "assist", "bonus_netti", "presenze",
                  "voto_medio", "minuti_totali", "xg_totale", "xa_totale",
                  "shots_totali", "presenze_titolare",
                  "ammonizioni_totali", "espulsioni_totali",
                  "rigori_segnati_totali", "rigori_sbagliati_totali",
                  "rigori_parati_totali"]

# Squadre italiane partecipanti alla fase a gironi/campionato della UEFA
# Champions League DI QUELLA STAGIONE (non i turni preliminari). Verificato
# via fetch diretto delle pagine Wikipedia "UEFA Champions League AAAA-AAAA
# (fase a gironi)" / "UEFA Champions League AAAA-AAAA" per ogni stagione,
# NON assunto un pattern fisso "sempre le prime 4 classificate dell'anno
# prima" - falso in più occasioni:
#  - 2015-16: solo Juventus e Roma (Napoli 2a e Inter 4a NON qualificate
#    quell'anno, la Roma nemmeno era nelle prime 2 classificate 2014-15
#    ma passò dai preliminari)
#  - 2016-17: solo Napoli e Juventus (2 squadre)
#  - 2017-18: Roma, Juventus, Napoli (Napoli passata dai preliminari)
#  - 2023-24: 4 squadre, le prime 4 classificate Serie A 2022-23 (Napoli,
#    Lazio, Inter, Milan) - coefficiente Italia 3° in Europa quell'anno,
#    NESSUN posto extra (verificato via Wikipedia "UEFA Champions League
#    2023-2024": Italia qualificata per 4 posti in fase a gironi)
#  - 2024-25: 5 squadre per posto extra dato dal ranking UEFA nazioni
#    (Atalanta, Inter, Milan, Juventus, Bologna)
SQUADRE_CHAMPIONS = {
    "2015-16": {"Juventus", "Roma"},
    "2016-17": {"Napoli", "Juventus"},
    "2017-18": {"Roma", "Juventus", "Napoli"},
    "2018-19": {"Juventus", "Napoli", "Roma", "Inter"},
    "2019-20": {"Juventus", "Atalanta", "Napoli", "Inter"},
    "2020-21": {"Juventus", "Inter", "Lazio", "Atalanta"},
    "2021-22": {"Milan", "Inter", "Atalanta", "Juventus"},
    "2022-23": {"Napoli", "Inter", "Milan", "Juventus"},
    "2023-24": {"Napoli", "Inter", "Lazio", "Milan"},
    "2024-25": {"Atalanta", "Inter", "Milan", "Juventus", "Bologna"},
    "2025-26": {"Napoli", "Inter", "Atalanta", "Juventus"},
    "2026-27": {"Inter", "Napoli", "Roma", "Como"},
}


def in_champions(squadra, stagione):
    return 1.0 if squadra in SQUADRE_CHAMPIONS.get(stagione, set()) else 0.0


def to_float(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def carica_presenze_valide():
    """Lista di dict con le presenze valide (fonte_voto=redazione, senza_voto!=True)."""
    presenze = []
    with open(VOTI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["fonte_voto"] != "redazione":
                continue
            if str(row["senza_voto"]) == "True":
                continue
            voto = to_float(row["voto"])
            fantavoto = to_float(row["fantavoto"])
            if voto is None or fantavoto is None:
                continue
            stagione = row["stagione"]
            if stagione not in STAGIONE_IDX:
                continue
            presenze.append({
                "stagione": stagione,
                "player_id": row["player_id"],
                "nome_giocatore": row["nome_giocatore"],
                "squadra_giocatore": row["squadra_giocatore"],
                "ruolo": row["ruolo"],
                "voto": voto,
                "fantavoto": fantavoto,
                "gol_fatti": to_float(row["gol_fatti"]) or 0.0,
                "assist": to_float(row["assist"]) or 0.0,
                # AGGIORNAMENTO v6 (richiesta Dario: "aggiungi cartellini e
                # rigori" come feature): campi booleani/contatori gia'
                # presenti in voti_storici_2015_2026.csv, semplicemente non
                # ancora aggregati per stagione finora.
                "ammonizione": 1.0 if str(row["ammonizione"]) == "True" else 0.0,
                "espulsione": 1.0 if str(row["espulsione"]) == "True" else 0.0,
                "rigori_segnati": to_float(row["rigori_segnati"]) or 0.0,
                "rigori_sbagliati": to_float(row["rigori_sbagliati"]) or 0.0,
                "rigori_parati": to_float(row["rigori_parati"]) or 0.0,
            })
    log.info("Presenze valide caricate: %d", len(presenze))
    return presenze


def carica_understat_per_stagione():
    """dict {(player_id_understat, stagione): {'minuti_totali':.., 'xg_totale':.., 'xa_totale':.., 'shots_totali':..}}
    ATTENZIONE: 'player_id' qui e' l'ID INTERNO di Understat, NON il
    player_id di fantacalcio.it - i due spazi ID sono diversi e NON
    condividono valori (verificato: solo ~323/2282 ID numerici
    coincidono per puro caso su range sovrapposto, match reale ~2.5% se
    usato come join diretto). Il bridge corretto e' player_name_mapping.csv
    (stagione, squadra, nome_fantacalcio) -> player_id_understat, usato in
    aggrega_per_giocatore_stagione() sotto - QUESTA funzione resta indicizzata
    per player_id_understat, e' aggrega_per_giocatore_stagione() che deve
    tradurre l'ID fantacalcio nell'ID Understat prima di interrogare questo
    dict, non il contrario."""
    agg = defaultdict(lambda: {"minuti_totali": 0.0, "xg_totale": 0.0, "xa_totale": 0.0,
                                 "shots_totali": 0.0, "presenze_titolare": 0.0})
    with open(UNDERSTAT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stagione = row["stagione"]
            player_id_understat = row["player_id"]
            if stagione not in STAGIONE_IDX or not player_id_understat:
                continue
            key = (player_id_understat, stagione)
            agg[key]["minuti_totali"] += to_float(row.get("time")) or 0.0
            agg[key]["xg_totale"] += to_float(row.get("xG")) or 0.0
            agg[key]["xa_totale"] += to_float(row.get("xA")) or 0.0
            agg[key]["shots_totali"] += to_float(row.get("shots")) or 0.0
            # 'Sub' flagga i subentrati (verificato: mediana 17 min vs 90
            # min delle altre posizioni) - ogni riga con position!='Sub' e'
            # una presenza da titolare (sceso in campo dal 1').
            if row.get("position") != "Sub":
                agg[key]["presenze_titolare"] += 1.0
    log.info("Aggregati Understat per (player_id_understat, stagione): %d combinazioni", len(agg))
    return agg


def carica_name_mapping():
    """dict {(stagione, squadra_fantacalcio, nome_fantacalcio): player_id_understat}
    - BRIDGE CORRETTO tra i due spazi ID (fix del bug di join scoperto in
    questa fase: il codice precedente faceva understat_agg.get((player_id,
    stagione)) usando direttamente il player_id di fantacalcio, che NON e'
    lo stesso ID di Understat - risultato: ~97.7% delle righe con
    xg_totale_lag1/xa_totale_lag1/shots_totali_lag1/minuti_totali_lag1 = 0.0
    per mancato match, non per assenza genuina di dati). Stessa fonte e
    stessa logica di join gia' corretta in build_feature_dataset.py
    (carica_name_mapping/trova_stats_understat), qui riusata a livello
    stagionale invece che a livello singola partita. Copertura verificata:
    98.7% delle combinazioni (player_id, stagione) valide trova un match
    (vs 2.5% del join diretto precedente)."""
    mapping = {}
    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["stagione"], row["squadra"], row["nome_fantacalcio"])
            mapping[key] = row["player_id_understat"]
    log.info("Caricata mappa nomi giocatore (bridge fantacalcio->Understat): %d coppie (stagione, squadra, nome)", len(mapping))
    return mapping


def norm_team_understat(name):
    return TEAM_NAME_MAP_UNDERSTAT.get(name, name)


def carica_classifica_finale():
    """dict {(stagione, squadra): {'punti_finale':.., 'posizione_finale':..}}
    prendendo la riga con la giornata massima (38) di ciascuna stagione
    - e' la classifica FINALE, nota per costruzione a fine di quella
    stagione (prima che inizi la successiva)."""
    per_key_giornata_max = {}
    with open(CLASSIFICA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stagione = row["stagione"]
            squadra = row["squadra"]
            giornata = int(row["giornata"])
            key = (stagione, squadra)
            if key not in per_key_giornata_max or giornata > per_key_giornata_max[key][0]:
                per_key_giornata_max[key] = (giornata, {
                    "punti_pre": to_float(row["punti_pre"]),
                    "posizione_pre": to_float(row["posizione_pre"]),
                })
    # NOTA: punti_pre/posizione_pre all'ultima giornata (38) sono calcolati
    # PRIMA della partita di giornata 38, cioè sono la classifica dopo 37
    # giornate, non la classifica ufficiale finale dopo tutte le 38. E'
    # un'approssimazione accettabile (differenza massima di una giornata su
    # 38, irrilevante come feature di contesto "forza squadra") - dichiarata
    # esplicitamente, non un bug.
    finale = {key: v[1] for key, v in per_key_giornata_max.items()}
    log.info("Classifica finale (approssimata a giornata 38 pre-partita) per %d combinazioni (stagione, squadra)", len(finale))
    return finale


def carica_quotazioni():
    """dict {(player_id, stagione): {'quotazione_iniziale':.., 'quotazione_attuale':.., 'fvm':..}}"""
    quot = {}
    with open(QUOTAZIONI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = row["player_id"]
            stagione = row["stagione"]
            quot[(player_id, stagione)] = {
                "quotazione_iniziale": to_float(row["quotazione_iniziale"]),
                "quotazione_attuale": to_float(row["quotazione_attuale"]),
                "fvm": to_float(row["fvm"]),
            }
    log.info("Quotazioni caricate per %d combinazioni (player_id, stagione)", len(quot))
    return quot


def carica_eta():
    """dict {(player_id, stagione): eta_al_1_agosto} - da
    eta_giocatori_storico_2015_2026.csv (scrape_eta_giocatori.py,
    Transfermarkt). File opzionale: se assente (script non ancora
    eseguito), tutte le eta' risultano None, dichiarato esplicitamente nel
    log, NON e' un errore bloccante."""
    if not ETA_PATH.exists():
        log.warning("File eta' giocatori non trovato (%s): eta_n1 sara' sempre None per questa run", ETA_PATH)
        return {}
    eta = {}
    with open(ETA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = to_float(row["eta_al_1_agosto"])
            if v is not None:
                eta[(row["player_id"], row["stagione"])] = v
    log.info("Eta' giocatori caricate per %d combinazioni (player_id, stagione)", len(eta))
    return eta


def carica_infortuni():
    """dict {player_id: [{'stagione':.., 'giorni_stop':..}, ...]} - una
    lista di episodi per giocatore (da
    scrape_infortuni_profilo_giocatori.py/Transfermarkt). Gli episodi con
    'stagione' vuota (fuori dal range 2015-2026, verificato: sono tutti
    infortuni anteriori al 2015) vengono scartati qui, non contano per
    nessuna feature. File opzionale: se assente, tutte le feature
    infortuni_* risultano 0/None, dichiarato nel log, non bloccante."""
    per_player = defaultdict(list)
    if not INFORTUNI_PATH.exists():
        log.warning("File infortuni non trovato (%s): infortuni_* saranno sempre 0/None per questa run", INFORTUNI_PATH)
        return per_player
    scartati_stagione_vuota = 0
    tot = 0
    with open(INFORTUNI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tot += 1
            stagione = row["stagione"]
            if not stagione or stagione not in STAGIONE_IDX:
                scartati_stagione_vuota += 1
                continue
            per_player[row["player_id"]].append({
                "stagione": stagione,
                "giorni_stop": to_float(row["giorni_stop"]) or 0.0,
            })
    log.info("Episodi infortunio caricati: %d totali, %d scartati (stagione fuori range 2015-2026, verificato: infortuni ante-2015), per %d player_id distinti",
              tot, scartati_stagione_vuota, len(per_player))
    return per_player


def carica_profilo():
    """dict {player_id: {'altezza_m':.., 'nazionalita':.., 'piede_dominante':..}}
    da scrape_infortuni_profilo_giocatori.py/Transfermarkt. Profilo
    "attuale" (non storico per stagione) - stessa semplificazione
    dichiarata nella docstring del modulo. File opzionale: se assente,
    tutte le feature profilo risultano None."""
    profilo = {}
    if not PROFILO_PATH.exists():
        log.warning("File profilo non trovato (%s): altezza_m/nazionalita/piede_dominante saranno sempre None per questa run", PROFILO_PATH)
        return profilo
    with open(PROFILO_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profilo[row["player_id"]] = {
                "altezza_m": to_float(row["altezza_m"]),
                "nazionalita": row["nazionalita"] or None,
                "piede_dominante": row["piede_dominante"] or None,
            }
    log.info("Profili fisici caricati per %d player_id", len(profilo))
    return profilo


def carica_forum_esperti():
    """dict {player_id: {colonna_forum: valore}} da
    scrape_forum_esperti_2026_27.py (schede squadra Gruppo Esperti,
    richiesto da Dario: "Consideriamo anche questi aspetti nell'analisi").
    Come per carica_profilo(): indicizzato per player_id PURO, non per
    (player_id, stagione) - e' uno SNAPSHOT UNICO pre-stagione-2026-27,
    non una serie storica (il forum non ha equivalenti per le stagioni
    passate). Righe senza player_id (match nome falliito, dichiarato nel
    log dello scraper) vengono ignorate qui, non forzate. File opzionale:
    se assente, tutte le feature forum_* risultano None per questa run."""
    forum = {}
    if not FORUM_ESPERTI_PATH.exists():
        log.warning("File forum esperti non trovato (%s): forum_* saranno sempre None per questa run", FORUM_ESPERTI_PATH)
        return forum
    righe_senza_player_id = 0
    duplicati = 0
    with open(FORUM_ESPERTI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = row.get("player_id")
            if not player_id:
                righe_senza_player_id += 1
                continue
            if player_id in forum:
                duplicati += 1
                continue
            forum[player_id] = {
                "titolarita_forum": to_float(row["titolarita_forum"]),
                "media_voto_forum": to_float(row["media_voto_forum"]),
                "salute_forum": to_float(row["salute_forum"]),
                "bonus_forum": to_float(row["bonus_forum"]),
                "consiglio_esperti_forum": to_float(row["consiglio_esperti_forum"]),
                "totale_forum": to_float(row["totale_forum"]),
            }
    log.info("Pagelle forum esperti caricate per %d player_id (righe senza player_id/non matchate scartate: %d)",
              len(forum), righe_senza_player_id)
    if duplicati:
        log.warning("player_id duplicati nel CSV forum (tenuta la prima occorrenza): %d", duplicati)
    return forum


def trova_understat_agg(name_mapping, understat_agg, stagione, pres):
    """Risolve l'ID Understat corretto per questo giocatore/stagione usando
    il bridge player_name_mapping.csv (nome_fantacalcio, squadra, stagione),
    poi recupera gli aggregati Understat per quell'ID - fix del bug di join
    diretto sul player_id (vedi nota in carica_understat_per_stagione).
    Prova tutte le combinazioni (nome, squadra) distinte comparse nella
    stagione (caso raro: cambio squadra a meta' stagione), prende la prima
    che trova un match. Nessun match -> dict vuoto (0.0 di default a valle,
    dichiarato esplicitamente come "non matchato", non silenziato)."""
    combinazioni = {(p["nome_giocatore"], p["squadra_giocatore"]) for p in pres}
    for nome, squadra in combinazioni:
        squadra_norm = norm_team_understat(squadra)
        player_id_understat = name_mapping.get((stagione, squadra_norm, nome))
        if player_id_understat:
            u = understat_agg.get((player_id_understat, stagione))
            if u:
                return u
    return {}


def aggrega_per_giocatore_stagione(presenze, understat_agg, name_mapping):
    """dict {(player_id, stagione): metriche aggregate}"""
    per_key = defaultdict(list)
    for p in presenze:
        per_key[(p["player_id"], p["stagione"])].append(p)

    aggregati = {}
    n_match_understat = 0
    for (player_id, stagione), pres in per_key.items():
        n = len(pres)
        fantamedia = sum(p["fantavoto"] for p in pres) / n
        voto_medio = sum(p["voto"] for p in pres) / n
        gol = sum(p["gol_fatti"] for p in pres)
        assist = sum(p["assist"] for p in pres)
        bonus_netti = sum(p["fantavoto"] - p["voto"] for p in pres)
        ammonizioni_totali = sum(p["ammonizione"] for p in pres)
        espulsioni_totali = sum(p["espulsione"] for p in pres)
        rigori_segnati_totali = sum(p["rigori_segnati"] for p in pres)
        rigori_sbagliati_totali = sum(p["rigori_sbagliati"] for p in pres)
        rigori_parati_totali = sum(p["rigori_parati"] for p in pres)
        u = trova_understat_agg(name_mapping, understat_agg, stagione, pres)
        if u:
            n_match_understat += 1
        aggregati[(player_id, stagione)] = {
            "fantamedia": fantamedia,
            "voto_medio": voto_medio,
            "gol": gol,
            "assist": assist,
            "bonus_netti": bonus_netti,
            "presenze": float(n),
            "minuti_totali": u.get("minuti_totali", 0.0),
            "xg_totale": u.get("xg_totale", 0.0),
            "xa_totale": u.get("xa_totale", 0.0),
            "shots_totali": u.get("shots_totali", 0.0),
            "presenze_titolare": u.get("presenze_titolare", 0.0),
            "ammonizioni_totali": ammonizioni_totali,
            "espulsioni_totali": espulsioni_totali,
            "rigori_segnati_totali": rigori_segnati_totali,
            "rigori_sbagliati_totali": rigori_sbagliati_totali,
            "rigori_parati_totali": rigori_parati_totali,
            "ruolo": max(set(p["ruolo"] for p in pres), key=lambda r: sum(1 for p in pres if p["ruolo"] == r)),
            "squadra_giocatore": pres[-1]["squadra_giocatore"],  # ultima nota nella stagione
            "nome_giocatore": pres[-1]["nome_giocatore"],
        }
    log.info("Aggregati (player_id, stagione) calcolati: %d", len(aggregati))
    log.info("Di questi, con match Understat riuscito (xg/xa/shots/minuti reali, non 0.0 di default): %d (%.1f%%)",
              n_match_understat, 100 * n_match_understat / len(aggregati) if aggregati else 0)
    return aggregati


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def std_or_none(values):
    """Deviazione standard (popolazione, ddof=0) delle stesse fino-3
    stagioni precedenti usate per _ma3 - richiesta da Dario ("considera
    anche la varianza oltre alle medie mobili"): un giocatore con
    fantamedia_ma3=6.5 costruita su 6.0/6.5/7.0 e' piu' "instabile" di uno
    con 6.4/6.5/6.6, stessa media ma varianza diversa - informazione persa
    usando solo la media. None se <2 valori disponibili (varianza non
    definita/poco significativa su 1 solo valore, non riempita a 0 per non
    confonderla con "davvero stabile")."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def trend_or_none(values_cronologiche):
    """Pendenza (slope) di una regressione lineare semplice y = a + b*x
    sulle stesse fino-3 stagioni precedenti usate per _ma3/_std3, con
    values_cronologiche in ordine DAL PIU' VECCHIO AL PIU' RECENTE (x=0,1,2..)
    - richiesta da Dario ("considera... il trend"): un giocatore andato
    5.5->6.0->6.5 (trend positivo) e uno andato 6.5->6.0->5.5 (trend
    negativo) hanno la stessa media (6.0) ma direzioni opposte, rilevante
    per capire se sta migliorando o peggiorando. None se <2 valori
    disponibili (una pendenza non e' definibile su un solo punto)."""
    vals = [v for v in values_cronologiche if v is not None]
    n = len(vals)
    if n < 2:
        return None
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(vals) / n
    num = sum((xs[i] - x_mean) * (vals[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def costruisci_righe(aggregati, classifica_finale, quotazioni, eta_map, infortuni_map, profilo_map, forum_map):
    per_player = defaultdict(dict)  # player_id -> {stagione: aggregato}
    for (player_id, stagione), agg in aggregati.items():
        per_player[player_id][stagione] = agg

    righe = []
    for player_id, per_stagione in per_player.items():
        stagioni_giocatore = sorted(per_stagione.keys(), key=lambda s: STAGIONE_IDX[s])

        for stagione_target in stagioni_giocatore:
            idx_target = STAGIONE_IDX[stagione_target]
            if idx_target == 0:
                # 2015-16: nessuna stagione precedente disponibile, non si
                # può costruire alcuna feature lag -> non genera una riga
                continue

            stagione_n1 = STAGIONI_STORICHE[idx_target - 1]
            agg_n1 = per_stagione.get(stagione_n1)
            if agg_n1 is None:
                # il giocatore non ha presenze valide nella stagione
                # immediatamente precedente: nessuna feature autoregressiva
                # affidabile -> non genera una riga (si esclude, non si
                # riempie con zero fittizio)
                continue

            target_agg = per_stagione[stagione_target]

            # media mobile fino a 3 stagioni precedenti disponibili (N-1,N-2,N-3)
            stagioni_prec = []
            for back in (1, 2, 3):
                idx_prec = idx_target - back
                if idx_prec < 0:
                    break
                s_prec = STAGIONI_STORICHE[idx_prec]
                if s_prec in per_stagione:
                    stagioni_prec.append(per_stagione[s_prec])

            riga = {
                "stagione_target": stagione_target,
                "player_id": player_id,
                "nome_giocatore": target_agg["nome_giocatore"],
                "ruolo": agg_n1["ruolo"],
                "squadra_giocatore_target": target_agg["squadra_giocatore"],
                "squadra_giocatore_n1": agg_n1["squadra_giocatore"],
                "cambio_squadra": 1.0 if target_agg["squadra_giocatore"] != agg_n1["squadra_giocatore"] else 0.0,
                "presenze_cumulate_carriera_n1": float(sum(
                    1 for s in stagioni_giocatore if STAGIONE_IDX[s] < idx_target
                )),
            }

            # target (5 espliciti + presenze di contesto, che ora e' anche
            # essa stessa un target esplicito - vedi AGGIORNAMENTO v4 sotto)
            riga["fantamedia_target"] = target_agg["fantamedia"]
            riga["gol_target"] = target_agg["gol"]
            riga["assist_target"] = target_agg["assist"]
            riga["bonus_netti_target"] = target_agg["bonus_netti"]
            riga["presenze_target"] = target_agg["presenze"]
            # AGGIORNAMENTO v4 (richiesta Dario: previsioni "voto" oltre a
            # fantamedia/gol/assist): voto_medio_target = media del "voto"
            # REDAZIONE (giudizio tecnico puro) sulla stagione N, DISTINTO
            # da fantamedia_target che include bonus/malus (gol, assist,
            # cartellini, ecc.). Era in COLONNE_VIETATE come rete di
            # sicurezza generica contro leakage accidentale finche' non era
            # un target dichiarato: ora e' un target esplicito voluto,
            # quindi rimosso da quell'insieme piu' sotto (stesso principio
            # gia' applicato a fantamedia_target/gol_target/ecc., che sono
            # anch'essi aggregati della stagione target ma dichiarati).
            riga["voto_medio_target"] = target_agg["voto_medio"]
            # AGGIORNAMENTO v4 (richiesta Dario: "partite titolari" nella
            # lista completa): presenze_titolare_target = numero di
            # presenze da titolare (position!='Sub' su Understat) nella
            # stagione N - possibile SOLO ora che il bug di join Understat
            # e' stato corretto (senza il fix, presenze_titolare sarebbe
            # quasi sempre 0.0 per lo stesso motivo di xg/xa/shots).
            riga["presenze_titolare_target"] = target_agg["presenze_titolare"]

            # feature lag1 (stagione N-1, componente autoregressiva diretta)
            for m in METRICHE_BASE:
                riga[f"{m}_lag1"] = agg_n1[m]

            # feature media mobile ultime fino a 3 stagioni precedenti
            for m in METRICHE_BASE:
                riga[f"{m}_ma3"] = mean_or_none([s[m] for s in stagioni_prec])

            # AGGIORNAMENTO v5 (richiesta Dario: "considera anche la
            # varianza oltre alle medie mobili e il trend"): stessa finestra
            # di stagioni_prec (fino a 3 precedenti), due nuove feature per
            # ciascuna metrica base - vedi std_or_none/trend_or_none per la
            # motivazione. stagioni_prec e' in ordine "piu' recente prima"
            # (back=1,2,3), va invertito per il trend (serve "piu' vecchio
            # prima" per una pendenza temporale con segno corretto: positivo
            # = in crescita verso la stagione target, negativo = in calo).
            stagioni_prec_cronologiche = list(reversed(stagioni_prec))
            for m in METRICHE_BASE:
                riga[f"{m}_std3"] = std_or_none([s[m] for s in stagioni_prec])
                riga[f"{m}_trend3"] = trend_or_none([s[m] for s in stagioni_prec_cronologiche])

            # feature media di carriera: TUTTE le stagioni precedenti
            # disponibili (non solo le ultime 3 come ma3) - richiesta da
            # Dario per non basarsi quasi solo sull'ultima stagione.
            stagioni_tutte_precedenti = [
                per_stagione[s] for s in stagioni_giocatore if STAGIONE_IDX[s] < idx_target
            ]
            for m in METRICHE_BASE:
                riga[f"{m}_career_mean"] = mean_or_none([s[m] for s in stagioni_tutte_precedenti])

            # eta' al 1 agosto della stagione N-1 (sempre pre-stagione-N per
            # costruzione). None se il giocatore non e' stato matchato alla
            # fonte Transfermarkt - dichiarato, non riempito a caso.
            riga["eta_n1"] = eta_map.get((player_id, stagione_n1))

            # flag Champions League: squadra N-1 (nota) e squadra target
            # (nota prima che la stagione N cominci, qualificazioni decise
            # a fine N-1 - non e' leakage, stesso principio di
            # quotazione_iniziale_target)
            riga["squadra_in_champions_n1"] = in_champions(agg_n1["squadra_giocatore"], stagione_n1)
            riga["squadra_in_champions_target"] = in_champions(target_agg["squadra_giocatore"], stagione_target)

            # contesto squadra: classifica finale della squadra N-1
            ctx_n1 = classifica_finale.get((stagione_n1, agg_n1["squadra_giocatore"]))
            riga["squadra_punti_finali_n1"] = ctx_n1["punti_pre"] if ctx_n1 else None
            riga["squadra_posizione_finale_n1"] = ctx_n1["posizione_pre"] if ctx_n1 else None

            # se il giocatore ha cambiato squadra, riporta anche il
            # contesto della NUOVA squadra nella stagione N-1 (forza della
            # squadra che lo accoglie, nota prima che N cominci)
            if riga["cambio_squadra"] == 1.0:
                ctx_new = classifica_finale.get((stagione_n1, target_agg["squadra_giocatore"]))
                riga["squadra_nuova_punti_finali_n1"] = ctx_new["punti_pre"] if ctx_new else None
                riga["squadra_nuova_posizione_finale_n1"] = ctx_new["posizione_pre"] if ctx_new else None
            else:
                riga["squadra_nuova_punti_finali_n1"] = None
                riga["squadra_nuova_posizione_finale_n1"] = None

            # quotazioni ufficiali: N-1 (doppiamente pre-stagione-N) e N
            # stessa (quotazione INIZIALE della stagione target, nota prima
            # che la stagione N cominci - non e' leakage, e' il prezzo
            # d'asta ufficiale pubblicato in anticipo)
            quot_n1 = quotazioni.get((player_id, stagione_n1))
            riga["quotazione_iniziale_n1"] = quot_n1["quotazione_iniziale"] if quot_n1 else None
            riga["quotazione_attuale_n1"] = quot_n1["quotazione_attuale"] if quot_n1 else None
            # FVM pubblicato in scala "/1000" -> riscalato /2 per un budget
            # d'asta a 500 crediti (richiesto da Dario: tenere FVM come
            # feature aggiuntiva, "previsione di spesa fatta da fantacalcio").
            # None esplicito se il dato non esiste (stagioni <2022-23),
            # nessun riempimento fittizio.
            fvm_n1_raw = quot_n1["fvm"] if quot_n1 else None
            riga["fvm_n1"] = (fvm_n1_raw / 2.0) if fvm_n1_raw is not None else None

            quot_target = quotazioni.get((player_id, stagione_target))
            riga["quotazione_iniziale_target"] = quot_target["quotazione_iniziale"] if quot_target else None

            # infortuni: episodi del giocatore con stagione == N-1 (per
            # costruzione, sempre nota a fine N-1, prima che N cominci) e
            # conteggio cumulato su TUTTE le stagioni < N (stesso principio
            # di _career_mean). 0 (non None) se il giocatore e' stato
            # scrappato ma non ha episodi in quella stagione - un'assenza
            # di infortuni e' un dato valido, non mancante. None solo se il
            # player_id non e' MAI stato trovato nello scraping (dichiarato
            # via infortuni_map.get(player_id) is None più sotto, gestito
            # implicitamente: lista vuota in entrambi i casi con questa
            # struttura dati - vedi nota nel log finale di main() per la
            # distinzione).
            episodi_giocatore = infortuni_map.get(player_id, [])
            episodi_n1 = [e for e in episodi_giocatore if e["stagione"] == stagione_n1]
            riga["infortuni_n1_count"] = float(len(episodi_n1))
            riga["infortuni_n1_giorni_totali"] = sum(e["giorni_stop"] for e in episodi_n1)
            episodi_career = [e for e in episodi_giocatore if STAGIONE_IDX.get(e["stagione"], 999) < idx_target]
            riga["infortuni_career_count"] = float(len(episodi_career))

            # profilo fisico: "attuale" al momento dello scraping, usato
            # anche per stagioni passate (semplificazione dichiarata nella
            # docstring del modulo). None se il player_id non e' stato
            # scrappato/matchato a Transfermarkt.
            prof = profilo_map.get(player_id)
            riga["altezza_m"] = prof["altezza_m"] if prof else None
            riga["nazionalita"] = prof["nazionalita"] if prof else None
            riga["piede_dominante"] = prof["piede_dominante"] if prof else None

            # forum Gruppo Esperti: valutazione QUALITATIVA soggettiva
            # pre-stagione, richiesta esplicitamente da Dario
            # ("Consideriamo anche questi aspetti nell'analisi"). Snapshot
            # UNICO disponibile solo per la stagione 2026-27 (il forum non
            # ha equivalenti per le stagioni passate) - popolato SOLO sulla
            # riga stagione_target=='2026-27', None su tutte le altre righe
            # storiche (MAI 0: 0 sarebbe un falso "nessun hype" per stagioni
            # in cui la feature semplicemente non esiste come concetto).
            # Non entra in METRICHE_BASE (non e' una serie storica
            # accumulabile per stagione) - stesso trattamento di
            # altezza_m/piede_dominante sopra.
            if stagione_target == "2026-27":
                forum = forum_map.get(player_id)
                riga["titolarita_forum"] = forum["titolarita_forum"] if forum else None
                riga["media_voto_forum"] = forum["media_voto_forum"] if forum else None
                riga["salute_forum"] = forum["salute_forum"] if forum else None
                riga["bonus_forum"] = forum["bonus_forum"] if forum else None
                riga["consiglio_esperti_forum"] = forum["consiglio_esperti_forum"] if forum else None
                riga["totale_forum"] = forum["totale_forum"] if forum else None
            else:
                riga["titolarita_forum"] = None
                riga["media_voto_forum"] = None
                riga["salute_forum"] = None
                riga["bonus_forum"] = None
                riga["consiglio_esperti_forum"] = None
                riga["totale_forum"] = None

            righe.append(riga)

    log.info("Righe (player_id, stagione_target) costruite: %d", len(righe))
    return righe


COLONNE_VIETATE = {
    # nessuna colonna con suffisso diverso da _lag1/_ma3/_career_mean/_n1/
    # _target (quotazione_iniziale_target, squadra_in_champions_target e,
    # da v4, voto_medio_target sono le eccezioni dichiarate: tutte
    # genuinamente note prima che la stagione target cominci OPPURE, nel
    # caso di voto_medio_target, un target esplicito voluto da Dario -
    # stesso principio gia' applicato a fantamedia_target/gol_target/ecc.)
    # calcolata dagli aggregati della stagione TARGET stessa deve esistere
    # nell'output, salvo i target espliciti.
    "minuti_totali_target", "xg_totale_target",
    "xa_totale_target", "shots_totali_target",
    "quotazione_attuale_target", "fvm_target", "eta_target",
    "infortuni_target_count", "infortuni_n_count", "altezza_target",
}


def verifica_anti_leakage(fieldnames):
    presenti = set(fieldnames) & COLONNE_VIETATE
    if presenti:
        log.error("VIOLAZIONE ANTI-LEAKAGE: colonne vietate presenti nell'output: %s", presenti)
        raise SystemExit(1)
    log.info("Verifica anti-leakage OK: nessuna colonna vietata presente tra le %d colonne finali", len(fieldnames))


def main():
    presenze = carica_presenze_valide()
    understat_agg = carica_understat_per_stagione()
    name_mapping = carica_name_mapping()
    classifica_finale = carica_classifica_finale()
    quotazioni = carica_quotazioni()
    eta_map = carica_eta()
    infortuni_map = carica_infortuni()
    profilo_map = carica_profilo()
    forum_map = carica_forum_esperti()

    aggregati = aggrega_per_giocatore_stagione(presenze, understat_agg, name_mapping)
    righe = costruisci_righe(aggregati, classifica_finale, quotazioni, eta_map, infortuni_map, profilo_map, forum_map)

    if not righe:
        log.error("Nessuna riga costruita, interrompo.")
        return

    fieldnames = list(righe[0].keys())
    verifica_anti_leakage(fieldnames)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(righe)

    log.info("Dataset scritto: %d righe, %d colonne, in %s", len(righe), len(fieldnames), OUT_PATH)

    # copertura per stagione target
    per_stagione_count = defaultdict(int)
    per_stagione_quot_n1 = defaultdict(int)
    per_stagione_quot_target = defaultdict(int)
    per_stagione_eta = defaultdict(int)
    per_stagione_champions = defaultdict(int)
    per_stagione_profilo = defaultdict(int)
    for r in righe:
        per_stagione_count[r["stagione_target"]] += 1
        if r["quotazione_iniziale_n1"] is not None:
            per_stagione_quot_n1[r["stagione_target"]] += 1
        if r["quotazione_iniziale_target"] is not None:
            per_stagione_quot_target[r["stagione_target"]] += 1
        if r["eta_n1"] is not None:
            per_stagione_eta[r["stagione_target"]] += 1
        if r["squadra_in_champions_target"] == 1.0:
            per_stagione_champions[r["stagione_target"]] += 1
        if r["altezza_m"] is not None:
            per_stagione_profilo[r["stagione_target"]] += 1
    for s in STAGIONI_STORICHE:
        if s in per_stagione_count:
            tot = per_stagione_count[s]
            log.info("  %s: %d righe, quot_n1=%d (%.1f%%), quot_target=%d (%.1f%%), eta_n1=%d (%.1f%%), profilo=%d (%.1f%%), in_champions_target=%d",
                      s, tot, per_stagione_quot_n1[s], 100 * per_stagione_quot_n1[s] / tot,
                      per_stagione_quot_target[s], 100 * per_stagione_quot_target[s] / tot,
                      per_stagione_eta[s], 100 * per_stagione_eta[s] / tot,
                      per_stagione_profilo[s], 100 * per_stagione_profilo[s] / tot,
                      per_stagione_champions[s])

    # statistiche infortuni (non per-stagione, aggregate sull'intero dataset)
    n_con_infortuni_n1 = sum(1 for r in righe if r["infortuni_n1_count"] > 0)
    n_con_infortuni_career = sum(1 for r in righe if r["infortuni_career_count"] > 0)
    log.info("Righe con almeno 1 infortunio in N-1: %d/%d (%.1f%%)", n_con_infortuni_n1, len(righe), 100 * n_con_infortuni_n1 / len(righe))
    log.info("Righe con almeno 1 infortunio in carriera (< N): %d/%d (%.1f%%)", n_con_infortuni_career, len(righe), 100 * n_con_infortuni_career / len(righe))

    # forum Gruppo Esperti: copertura ESCLUSIVAMENTE sulle righe
    # stagione_target=='2026-27' (per costruzione None altrove, atteso e
    # dichiarato, non un errore) - dettaglio a parte per non confondere il
    # riepilogo per-stagione sopra, dove sarebbe quasi sempre 0.
    righe_2026_27 = [r for r in righe if r["stagione_target"] == "2026-27"]
    if righe_2026_27:
        n_con_forum = sum(1 for r in righe_2026_27 if r["totale_forum"] is not None)
        log.info("Righe 2026-27 con pagelle forum esperti disponibili: %d/%d (%.1f%%) - None su tutte le altre stagioni per costruzione (snapshot pre-2026-27, non serie storica)",
                  n_con_forum, len(righe_2026_27), 100 * n_con_forum / len(righe_2026_27))
    else:
        log.warning("Nessuna riga stagione_target=='2026-27' nel dataset: verificare quotazioni_fantacalcio_storico_2015_2026.csv")


if __name__ == "__main__":
    main()
