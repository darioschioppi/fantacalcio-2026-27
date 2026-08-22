#!/usr/bin/env python3
"""
Genera giornata1_data.js (root del sito GitHub Pages) da
work/data/previsioni_serie_a_2026_27_giornata1.csv, richiesto da Dario
per pubblicare sul sito i dati delle "schede partita" giornata 1
(vedi Aggiornamento v8 in work/README.md).

A differenza di qualitativo_data.js/mercato_vs_modello_data.js (generati
con un processo semi-manuale one-off, come indicato nei loro header),
questo file JS e' generato SEMPRE dallo stesso CSV di origine via questo
script - va ri-eseguito ogni volta che
merge_schede_partita_giornata1.py produce un nuovo
previsioni_serie_a_2026_27_giornata1.csv (es. per una giornata diversa,
se in futuro si estende l'analisi ad altre giornate).

Filtro: solo le righe con dati di giornata1 effettivamente disponibili
(giornata1_squadra_casa non vuoto) - un giocatore matchato sulla scheda
partita ma senza NE' ballottaggio NE' indisponibilita' dichiarati ha
comunque un voto_consigliato_giornata1 (copre tutta la rosa, non solo i
casi dubbi) quindi resta incluso; solo indice_rischio_giornata1 puo'
essere vuoto quando manca il segnale per calcolarlo (nessuna invenzione
di dati, stesso principio del resto del progetto).

Uso:
  python3 generate_giornata1_data_js.py
"""
import csv
import json
import logging
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH = DATA_DIR / "previsioni_serie_a_2026_27_giornata1.csv"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "giornata1_data.js"

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO %(message)s")
log = logging.getLogger("generate_giornata1_data_js")


def to_bool_o_none(v):
    if v in (None, ""):
        return None
    return v == "True"


def to_float_o_none(v):
    if v in (None, ""):
        return None
    return float(v)


def main():
    if not IN_PATH.exists():
        log.error("File non trovato: %s - esegui prima merge_schede_partita_giornata1.py", IN_PATH)
        raise SystemExit(1)

    with open(IN_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        righe = list(reader)

    record = []
    for r in righe:
        if not r.get("giornata1_squadra_casa"):
            continue
        record.append({
            "squadra": r["squadra"],
            "nome": r["nome"],
            "ruolo": r["ruolo"],
            "squadra_casa": r["giornata1_squadra_casa"],
            "squadra_ospite": r["giornata1_squadra_ospite"],
            "data_ora": r["giornata1_data_ora"],
            "titolare_previsto": to_bool_o_none(r["titolare_previsto_giornata1"]),
            "ballottaggio_pct": to_float_o_none(r["ballottaggio_pct_giornata1"]),
            "indisponibile": to_bool_o_none(r["indisponibile_giornata1"]) or False,
            "motivo_indisponibilita": r["motivo_indisponibilita_giornata1"] or None,
            "voto_consigliato": int(r["voto_consigliato_giornata1"]) if r["voto_consigliato_giornata1"] else None,
            "commento_voto": r["commento_voto_giornata1"] or None,
            "rigorista": r["rigorista_giornata1"] == "True",
            "tira_punizioni": r["tira_punizioni_giornata1"] == "True",
            "tira_angoli": r["tira_angoli_giornata1"] == "True",
            "pred_presenze_titolare_previste": to_float_o_none(r["pred_presenze_titolare_previste"]),
            "indice_rischio": to_float_o_none(r["indice_rischio_giornata1"]),
            "categoria_rischio": r["categoria_rischio_giornata1"] or None,
        })

    record.sort(key=lambda x: (x["data_ora"], x["squadra_casa"], x["squadra"] != x["squadra_casa"], x["nome"]))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("// Schede partita giornata 1 2026-27 (forum Gruppo Esperti): ballottaggi, indisponibili,\n")
        f.write("// voto consigliato 1-5 e nuovo indice_rischio_giornata1 (fuzzy Mamdani) per ogni giocatore.\n")
        f.write("// Generato automaticamente da work/scripts/generate_giornata1_data_js.py\n")
        f.write("// a partire da work/data/previsioni_serie_a_2026_27_giornata1.csv - vedi work/README.md v8.\n")
        f.write("const GIORNATA1_DATA = ")
        f.write(json.dumps(record, ensure_ascii=False, indent=None, separators=(",", ":")))
        f.write(";\n")

    log.info("Output scritto in %s (%d giocatori, 10 partite)", OUT_PATH, len(record))


if __name__ == "__main__":
    main()
