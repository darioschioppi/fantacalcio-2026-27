#!/usr/bin/env python3
"""
Costruisce una mappa di corrispondenza nomi giocatore tra fantacalcio.it
(nome breve, tipicamente il cognome, es. "Reina") e Understat
(nome completo con accenti, es. "José Reina"), necessaria per integrare le
statistiche individuali Understat (xG/xA/tiri) nel dataset voti.

STRATEGIA: matching LOCALE per partita, non globale. Per ogni partita
(stagione + data + squadra) si prendono la rosa fantacalcio.it (nomi brevi)
e la rosa Understat (nomi completi) di QUELLA partita, e si appaiano i nomi
tra questi due insiemi ristretti (tipicamente 14-25 giocatori a squadra a
partita). Questo evita ambiguità tra omonimi che si verificherebbero con un
matching su tutto il dataset (es. più "Sandro" in stagioni diverse).

Algoritmo di match per singola coppia (nome_breve, nome_completo):
  1. normalizza entrambi: minuscolo, accenti rimossi, punteggiatura rimossa
  2. match esatto se nome_breve normalizzato == uno dei token del nome
     completo, o coincide con l'intero nome completo
  3. altrimenti, match se nome_breve è l'ultimo token del nome completo
     (il caso più comune: fantacalcio.it usa il cognome)
  4. altrimenti, fallback a match "contains" (nome_breve è sottostringa del
     nome completo o viceversa) - copre iniziali/abbreviazioni parziali
Ogni giocatore Understat della rosa può essere assegnato a UN SOLO nome
fantacalcio (matching greedy 1-a-1 con punteggio, per evitare che due nomi
brevi diversi finiscano sullo stesso giocatore Understat).

Output:
  work/data/player_name_mapping.csv
      colonne: stagione, squadra, nome_fantacalcio, player_understat,
               player_id_understat, match_type, n_partite_confermate
  work/data/player_name_mapping_log.txt (statistiche di copertura/ambiguità)

Uso:
  python3 build_player_name_mapping.py
"""
import csv
import re
import logging
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOTI_PATH = DATA_DIR / "voti_storici_2015_2026.csv"
UNDERSTAT_PATH = DATA_DIR / "understat_player_match_stats_storico_2015_2026.csv"
OUT_PATH = DATA_DIR / "player_name_mapping.csv"
LOG_PATH = DATA_DIR / "player_name_mapping_log.txt"

TEAM_NAME_MAP = {
    "SPAL": "SPAL 2013",
    "Milan": "AC Milan",
    "Parma": "Parma Calcio 1913",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("name_mapping")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s):
    if not s:
        return ""
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_team(name):
    return TEAM_NAME_MAP.get(name, name)


def match_score(nome_breve_norm, nome_completo_norm):
    """Ritorna (score, match_type) più alto è meglio, None se nessun match plausibile."""
    tokens = nome_completo_norm.split()
    if nome_breve_norm == nome_completo_norm:
        return (3, "esatto")
    if tokens and nome_breve_norm == tokens[-1]:
        return (2, "cognome")
    if nome_breve_norm in tokens:
        return (2, "token")
    # abbreviazioni con iniziale, es. "Balogh N." -> norm "balogh n"
    breve_tokens = nome_breve_norm.split()
    if len(breve_tokens) >= 2 and breve_tokens[0] in tokens:
        # controlla se l'iniziale corrisponde a un token del nome completo
        iniziale = breve_tokens[-1].rstrip(".")
        if any(t.startswith(iniziale) for t in tokens if t != breve_tokens[0]):
            return (1, "cognome+iniziale")
    if nome_completo_norm.endswith(nome_breve_norm) or nome_breve_norm.endswith(nome_completo_norm):
        return (1, "suffix")
    if nome_breve_norm in nome_completo_norm or nome_completo_norm in nome_breve_norm:
        return (0, "contains")
    return None


def carica_rose_voti():
    """dict {(stagione, data, squadra_norm): set(nome_giocatore)}"""
    rose = defaultdict(set)
    with open(VOTI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["stagione"], row["data"], norm_team(row["squadra_giocatore"]))
            rose[key].add(row["nome_giocatore"])
    return rose


def carica_rose_understat():
    """dict {(stagione, data, squadra): list of (player, player_id)}"""
    rose = defaultdict(list)
    with open(UNDERSTAT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = (row["match_date"] or "")[:10]  # solo data, no ora
            key = (row["stagione"], data, row["team_title"])
            rose[key].append((row["player"], row["player_id"]))
    return rose


def trova_rosa_understat(rose_understat, stagione, data, squadra):
    """Cerca la rosa Understat per (stagione, data, squadra); se non trovata
    esattamente, prova ±1 giorno per assorbire lo scarto di fuso orario
    (Understat salva l'orario UTC, per partite serali la data locale può
    differire di un giorno da quella UTC)."""
    key = (stagione, data, squadra)
    if key in rose_understat:
        return rose_understat[key]
    from datetime import date, timedelta
    y, m, d = (int(x) for x in data.split("-"))
    base = date(y, m, d)
    for delta in (1, -1):
        alt_data = (base + timedelta(days=delta)).isoformat()
        alt_key = (stagione, alt_data, squadra)
        if alt_key in rose_understat:
            return rose_understat[alt_key]
    return None


def main():
    rose_voti = carica_rose_voti()
    rose_understat = carica_rose_understat()

    log.info("Partite/squadre voti: %d, Partite/squadre understat: %d", len(rose_voti), len(rose_understat))

    # accumulo dei match per (stagione, squadra, nome_fantacalcio) -> Counter(player_understat)
    accumulo = defaultdict(Counter)
    accumulo_id = {}  # (stagione, squadra, nome_fantacalcio, player_understat) -> player_id
    n_match_partite = 0
    n_partite_voti_totali = len(rose_voti)
    partite_non_trovate_in_understat = 0

    for key, nomi_brevi in rose_voti.items():
        stagione, data, squadra = key
        rosa_understat = trova_rosa_understat(rose_understat, stagione, data, squadra)
        if not rosa_understat:
            partite_non_trovate_in_understat += 1
            continue
        n_match_partite += 1

        nomi_brevi_norm = {n: norm(n) for n in nomi_brevi}
        rosa_understat_norm = [(p, pid, norm(p)) for p, pid in rosa_understat]

        usati_understat = set()
        # ordina i possibili match per punteggio decrescente, greedy assignment
        candidati = []
        for nome_breve, nb_norm in nomi_brevi_norm.items():
            for p, pid, p_norm in rosa_understat_norm:
                res = match_score(nb_norm, p_norm)
                if res:
                    score, mtype = res
                    candidati.append((score, nome_breve, p, pid, mtype))
        candidati.sort(key=lambda c: -c[0])

        usati_brevi = set()
        for score, nome_breve, p, pid, mtype in candidati:
            if nome_breve in usati_brevi or p in usati_understat:
                continue
            usati_brevi.add(nome_breve)
            usati_understat.add(p)
            accumulo[(stagione, squadra, nome_breve)][p] += 1
            accumulo_id[(stagione, squadra, nome_breve, p)] = pid

    log.info("Partite con rosa trovata in entrambe le fonti: %d/%d", n_match_partite, n_partite_voti_totali)
    log.info("Partite/squadra voti senza corrispondenza in understat: %d", partite_non_trovate_in_understat)

    # scegli, per ogni (stagione, squadra, nome_breve), il player_understat più frequente
    rows = []
    n_ambigui = 0
    for (stagione, squadra, nome_breve), counter in accumulo.items():
        player_understat, n_conferme = counter.most_common(1)[0]
        if len(counter) > 1:
            n_ambigui += 1
        pid = accumulo_id.get((stagione, squadra, nome_breve, player_understat))
        rows.append({
            "stagione": stagione,
            "squadra": squadra,
            "nome_fantacalcio": nome_breve,
            "player_understat": player_understat,
            "player_id_understat": pid,
            "n_partite_confermate": n_conferme,
        })

    log.info("Coppie (stagione, squadra, nome_fantacalcio) mappate: %d", len(rows))
    log.info("Di cui con più di un candidato Understat diverso nel corso della stagione (potenziale ambiguità): %d", n_ambigui)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["stagione", "squadra", "nome_fantacalcio", "player_understat", "player_id_understat", "n_partite_confermate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    log.info("Mappa scritta in %s", OUT_PATH)


if __name__ == "__main__":
    main()
