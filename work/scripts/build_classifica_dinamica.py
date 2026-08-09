#!/usr/bin/env python3
"""
Ricostruisce, per ogni (stagione, giornata, squadra), la classifica
PRE-PARTITA: punti/vittorie/pareggi/sconfitte/gol fatti/gol subiti/
differenza reti/posizione calcolati SOLO sulle partite già giocate nelle
giornate STRETTAMENTE PRECEDENTI a quella indicata (mai includendo la
partita della giornata stessa, per evitare data leakage nel modello ML:
la feature deve rappresentare cosa si sapeva PRIMA di quella partita).

Calcolo interamente locale, nessuna chiamata di rete: usa solo i risultati
partita-per-partita già presenti in work/data/voti_storici_2015_2026.csv
(colonne stagione, giornata, match_id, squadra_casa, squadra_ospite,
gol_casa, gol_ospite), deduplicati per match_id (ogni partita compare più
volte nel CSV voti, una riga per giocatore/fonte).

Per la giornata 1 di ogni stagione, la classifica pre-partita è ovviamente
a zero per tutte le squadre (nessuna partita ancora giocata).

Regola di ordinamento classifica: punti desc, poi differenza reti desc, poi
gol fatti desc (criterio semplificato standard; non replica eventuali
regolamenti speciali storici su classifiche avulse in caso di pari merito
su più squadre - irrilevante per l'uso come feature ML, che necessita solo
di un ranking plausibile, non della classifica UFFICIALE esatta in casi
limite di parità totale).

Output:
  work/data/classifica_dinamica_storico_2015_2026.csv
  colonne: stagione, giornata, squadra, punti_pre, vittorie_pre,
           pareggi_pre, sconfitte_pre, gol_fatti_pre, gol_subiti_pre,
           diff_reti_pre, posizione_pre

Uso:
  python3 build_classifica_dinamica.py
"""
import csv
import logging
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
OUT_PATH = DATA_DIR / "classifica_dinamica_storico_2015_2026.csv"
LOG_PATH = DATA_DIR / "classifica_dinamica_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("classifica_dinamica")


def carica_partite_univoche():
    """Ritorna dict {stagione: list of (giornata:int, squadra_casa, squadra_ospite, gol_casa:float, gol_ospite:float)}
    deduplicando per (stagione, match_id)."""
    partite_per_stagione = defaultdict(list)
    visti = set()
    with open(VOTI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["stagione"], row["match_id"])
            if key in visti:
                continue
            visti.add(key)
            try:
                giornata = int(row["giornata"])
                gol_casa = float(row["gol_casa"]) if row["gol_casa"] not in (None, "") else None
                gol_ospite = float(row["gol_ospite"]) if row["gol_ospite"] not in (None, "") else None
            except (ValueError, TypeError):
                continue
            if gol_casa is None or gol_ospite is None:
                continue
            partite_per_stagione[row["stagione"]].append(
                (giornata, row["squadra_casa"], row["squadra_ospite"], gol_casa, gol_ospite)
            )
    return partite_per_stagione


def calcola_classifica_dinamica(partite_per_stagione):
    """Ritorna lista di righe output."""
    righe_out = []

    for stagione, partite in partite_per_stagione.items():
        partite.sort(key=lambda p: p[0])  # ordina per giornata

        giornate_presenti = sorted(set(p[0] for p in partite))
        squadre = set()
        for _, casa, ospite, _, _ in partite:
            squadre.add(casa)
            squadre.add(ospite)

        # stato cumulato per squadra: punti, V, N, P, gf, gs
        stato = {sq: {"punti": 0, "V": 0, "N": 0, "P": 0, "gf": 0, "gs": 0} for sq in squadre}

        for giornata in giornate_presenti:
            # 1. Scrivi la classifica PRE-partita per questa giornata (stato accumulato fino a giornata-1)
            classifica = []
            for sq, s in stato.items():
                diff_reti = s["gf"] - s["gs"]
                classifica.append((sq, s["punti"], diff_reti, s["gf"], s))
            # ordina: punti desc, diff_reti desc, gol fatti desc
            classifica.sort(key=lambda c: (-c[1], -c[2], -c[3]))

            for pos, (sq, punti, diff_reti, gf, s) in enumerate(classifica, start=1):
                righe_out.append({
                    "stagione": stagione,
                    "giornata": giornata,
                    "squadra": sq,
                    "punti_pre": s["punti"],
                    "vittorie_pre": s["V"],
                    "pareggi_pre": s["N"],
                    "sconfitte_pre": s["P"],
                    "gol_fatti_pre": s["gf"],
                    "gol_subiti_pre": s["gs"],
                    "diff_reti_pre": s["gf"] - s["gs"],
                    "posizione_pre": pos,
                })

            # 2. Aggiorna lo stato con i risultati di QUESTA giornata (diventerà "pre" per la prossima)
            partite_giornata = [p for p in partite if p[0] == giornata]
            for _, casa, ospite, gol_casa, gol_ospite in partite_giornata:
                stato[casa]["gf"] += gol_casa
                stato[casa]["gs"] += gol_ospite
                stato[ospite]["gf"] += gol_ospite
                stato[ospite]["gs"] += gol_casa
                if gol_casa > gol_ospite:
                    stato[casa]["punti"] += 3
                    stato[casa]["V"] += 1
                    stato[ospite]["P"] += 1
                elif gol_casa < gol_ospite:
                    stato[ospite]["punti"] += 3
                    stato[ospite]["V"] += 1
                    stato[casa]["P"] += 1
                else:
                    stato[casa]["punti"] += 1
                    stato[ospite]["punti"] += 1
                    stato[casa]["N"] += 1
                    stato[ospite]["N"] += 1

        log.info("%s: %d giornate, %d squadre, classifica pre-partita calcolata", stagione, len(giornate_presenti), len(squadre))

    return righe_out


def main():
    partite_per_stagione = carica_partite_univoche()
    log.info("Stagioni caricate: %s", sorted(partite_per_stagione.keys()))
    for stagione, partite in partite_per_stagione.items():
        log.info("%s: %d partite univoche", stagione, len(partite))

    righe_out = calcola_classifica_dinamica(partite_per_stagione)

    fieldnames = [
        "stagione", "giornata", "squadra", "punti_pre", "vittorie_pre",
        "pareggi_pre", "sconfitte_pre", "gol_fatti_pre", "gol_subiti_pre",
        "diff_reti_pre", "posizione_pre",
    ]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in righe_out:
            writer.writerow(r)

    log.info("Scritte %d righe in %s", len(righe_out), OUT_PATH)


if __name__ == "__main__":
    main()
