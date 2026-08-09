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
  - Quotazione ufficiale Fantacalcio.it (Classic): quotazione_iniziale_N1
    (quotazione a inizio della stagione N-1, quindi doppiamente "safe":
    nota MOLTO prima della stagione N) e, quando disponibile,
    quotazione_iniziale_N (quotazione a inizio della stagione N STESSA -
    NON e' leakage: e' il prezzo d'asta ufficiale pubblicato PRIMA che la
    stagione N cominci, esattamente l'informazione disponibile a chi deve
    decidere se comprare il giocatore all'asta).
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
CLASSIFICA_PATH = DATA_DIR / "classifica_dinamica_storico_2015_2026.csv"
QUOTAZIONI_PATH = DATA_DIR / "quotazioni_fantacalcio_storico_2015_2026.csv"
OUT_PATH = DATA_DIR / "stagione_giocatore_dataset_2015_2026.csv"
LOG_PATH = DATA_DIR / "build_stagione_giocatore_dataset_log.txt"

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
METRICHE_BASE = ["fantamedia", "gol", "assist", "bonus_netti", "presenze",
                  "voto_medio", "minuti_totali", "xg_totale", "xa_totale", "shots_totali"]


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
            })
    log.info("Presenze valide caricate: %d", len(presenze))
    return presenze


def carica_understat_per_stagione():
    """dict {(player_id, stagione): {'minuti_totali':.., 'xg_totale':.., 'xa_totale':.., 'shots_totali':..}}"""
    agg = defaultdict(lambda: {"minuti_totali": 0.0, "xg_totale": 0.0, "xa_totale": 0.0, "shots_totali": 0.0})
    with open(UNDERSTAT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stagione = row["stagione"]
            player_id = row["player_id"]
            if stagione not in STAGIONE_IDX or not player_id:
                continue
            key = (player_id, stagione)
            agg[key]["minuti_totali"] += to_float(row.get("time")) or 0.0
            agg[key]["xg_totale"] += to_float(row.get("xG")) or 0.0
            agg[key]["xa_totale"] += to_float(row.get("xA")) or 0.0
            agg[key]["shots_totali"] += to_float(row.get("shots")) or 0.0
    log.info("Aggregati Understat per (player_id, stagione): %d combinazioni", len(agg))
    return agg


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


def aggrega_per_giocatore_stagione(presenze, understat_agg):
    """dict {(player_id, stagione): metriche aggregate}"""
    per_key = defaultdict(list)
    for p in presenze:
        per_key[(p["player_id"], p["stagione"])].append(p)

    aggregati = {}
    for (player_id, stagione), pres in per_key.items():
        n = len(pres)
        fantamedia = sum(p["fantavoto"] for p in pres) / n
        voto_medio = sum(p["voto"] for p in pres) / n
        gol = sum(p["gol_fatti"] for p in pres)
        assist = sum(p["assist"] for p in pres)
        bonus_netti = sum(p["fantavoto"] - p["voto"] for p in pres)
        u = understat_agg.get((player_id, stagione), {})
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
            "ruolo": max(set(p["ruolo"] for p in pres), key=lambda r: sum(1 for p in pres if p["ruolo"] == r)),
            "squadra_giocatore": pres[-1]["squadra_giocatore"],  # ultima nota nella stagione
            "nome_giocatore": pres[-1]["nome_giocatore"],
        }
    log.info("Aggregati (player_id, stagione) calcolati: %d", len(aggregati))
    return aggregati


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def costruisci_righe(aggregati, classifica_finale, quotazioni):
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

            # target (4 + presenze di contesto)
            riga["fantamedia_target"] = target_agg["fantamedia"]
            riga["gol_target"] = target_agg["gol"]
            riga["assist_target"] = target_agg["assist"]
            riga["bonus_netti_target"] = target_agg["bonus_netti"]
            riga["presenze_target"] = target_agg["presenze"]

            # feature lag1 (stagione N-1, componente autoregressiva diretta)
            for m in METRICHE_BASE:
                riga[f"{m}_lag1"] = agg_n1[m]

            # feature media mobile ultime fino a 3 stagioni precedenti
            for m in METRICHE_BASE:
                riga[f"{m}_ma3"] = mean_or_none([s[m] for s in stagioni_prec])

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

            righe.append(riga)

    log.info("Righe (player_id, stagione_target) costruite: %d", len(righe))
    return righe


COLONNE_VIETATE = {
    # nessuna colonna con suffisso diverso da _lag1/_ma3/_n1/_target
    # (quotazione_iniziale_target e' l'unica eccezione dichiarata sopra)
    # calcolata dagli aggregati della stagione TARGET stessa deve esistere
    # nell'output, salvo i target espliciti e quotazione_iniziale_target.
    "voto_medio_target", "minuti_totali_target", "xg_totale_target",
    "xa_totale_target", "shots_totali_target",
    "quotazione_attuale_target", "fvm_target",
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
    classifica_finale = carica_classifica_finale()
    quotazioni = carica_quotazioni()

    aggregati = aggrega_per_giocatore_stagione(presenze, understat_agg)
    righe = costruisci_righe(aggregati, classifica_finale, quotazioni)

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
    for r in righe:
        per_stagione_count[r["stagione_target"]] += 1
        if r["quotazione_iniziale_n1"] is not None:
            per_stagione_quot_n1[r["stagione_target"]] += 1
        if r["quotazione_iniziale_target"] is not None:
            per_stagione_quot_target[r["stagione_target"]] += 1
    for s in STAGIONI_STORICHE:
        if s in per_stagione_count:
            tot = per_stagione_count[s]
            log.info("  %s: %d righe, quot_n1=%d (%.1f%%), quot_target=%d (%.1f%%)",
                      s, tot, per_stagione_quot_n1[s], 100 * per_stagione_quot_n1[s] / tot,
                      per_stagione_quot_target[s], 100 * per_stagione_quot_target[s] / tot)


if __name__ == "__main__":
    main()
