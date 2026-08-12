#!/usr/bin/env python3
"""
Scraper del commento QUALITATIVO in linguaggio libero del forum Gruppo
Esperti (forum.gruppoesperti.it), a completamento di
scrape_forum_esperti_2026_27.py.

CONTESTO: quello scraper cattura solo il blocco NUMERICO strutturato per
ogni giocatore (Titolarità/Media voto/Salute/Bonus/Consiglio Esperti/
TOTALE, via BLOCCO_RE). Ispezionando dal vivo l'HTML del topic (verificato
su Atalanta t=232668 e Milan t=232671, struttura identica), OGNI topic
squadra contiene anche, DOPO i blocchi di ruolo (Portieri/Difensori/
Centrocampisti/Attaccanti/Rigoristi/Calci piazzati), tre sezioni di
analisi discorsiva delimitate da immagini-separatore
<img src=".../NOMESEZIONE" class="postimage">:

  CONSIGLIATI          -> giocatori che il forum consiglia di prendere
  POSSIBILI.SORPRESE   -> giocatori indicati come possibile sorpresa
  SCONSIGLIATI         -> giocatori che il forum sconsiglia

(dopo SCONSIGLIATI arriva PROSPETTO.PRIMAVERA, che NON e' analisi sui
giocatori di prima squadra - usato solo come terminatore).

Ogni paragrafo dentro queste sezioni ha il formato (verificato identico
su Atalanta/Milan):
  - <strong class="text-strong"><strong style="color:#HEXCOLOR">
      <span style="font-size:150%;line-height:116%">NOME:</span>
    </strong></strong> <strong class="text-strong">TESTO LIBERO...</strong>
Il colore e' semanticamente legato alla sezione (verificato: verde
259e2c=CONSIGLIATI, arancio ff8000=POSSIBILI.SORPRESE, rosso f80000=
SCONSIGLIATI) ma per robustezza determiniamo la categoria dalla POSIZIONE
del blocco rispetto ai marcatori immagine, non dal colore (piu' fragile
se il forum cambia palette).

Questo e' testo libero (scouting soggettivo, no schema fisso) -> non ha
senso un parsing strutturato oltre a nome+testo. La SINTESI del
contenuto (quello richiesto da Dario nel report) e' un passo successivo
affidato a un LLM, non a questo scraper.

MATCHING NOME -> player_id: stessa strategia (norm/match_score/
MIN_MATCH_SCORE) di scrape_forum_esperti_2026_27.py, ristretto alla
squadra corrente.

Output:
  work/data/forum_esperti_qualitativo_2026_27.csv
    squadra, categoria, nome_forum, analisi_forum, player_id
  work/data/scrape_forum_esperti_qualitativo_log.txt

Uso:
  python3 scrape_forum_esperti_qualitativo_2026_27.py
"""
import csv
import html as html_lib
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

from scrape_forum_esperti_2026_27 import (
    HEADERS,
    SQUADRA_TO_TOPIC_ID,
    carica_quotazioni_2026_27,
    get_con_retry,
    match_score,
    norm,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "forum_esperti_qualitativo_2026_27.csv"
LOG_PATH = DATA_DIR / "scrape_forum_esperti_qualitativo_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("scrape_forum_esperti_qualitativo")

MIN_MATCH_SCORE = 1

# Marcatori-immagine che delimitano le sezioni, nell'ordine in cui
# appaiono nella pagina (verificato dal vivo su Atalanta e Milan).
SEZIONI_ORDINE = ["CONSIGLIATI", "POSSIBILI.SORPRESE", "SCONSIGLIATI", "PROSPETTO.PRIMAVERA"]
CATEGORIA_PER_SEZIONE = {
    "CONSIGLIATI": "CONSIGLIATO",
    "POSSIBILI.SORPRESE": "POSSIBILE SORPRESA",
    "SCONSIGLIATI": "SCONSIGLIATO",
}

# Alcuni marcatori-immagine hanno un suffisso testuale nell'URL (es.
# ".../H7HVQNy.png/CONSIGLIATI"), altri no (es. una sotto-sezione
# "multiruolo" senza nome vista su Atalanta, ".../9Q5miXD.png" nudo).
# Per determinare la FINE di una sezione serve fermarsi al PROSSIMO
# marcatore-immagine qualsiasi, non solo a quelli con nome noto -
# altrimenti un marcatore anonimo dopo SCONSIGLIATI (che introduce
# contenuto NON pertinente, es. giocatori multiruolo) verrebbe incluso
# per errore nel segmento SCONSIGLIATI (bug scoperto dal vivo su
# Atalanta: EDERSON (M/C)/AHANOR (B/DS/E) erroneamente catturati come
# "SCONSIGLIATO" mentre sono in una sezione successiva senza nome).
MARKER_ANY_RE = re.compile(r'<img src="[^"]*\.(?:png|jpe?g|gif)(?:/([A-Za-z.]+))?" class="postimage"')

# Blocco "- NOME: testo libero" dentro una sezione. Il nome e' racchiuso
# nel wrapper colore, che nella pratica e' <strong style="color:...">
# oppure <span style="color:...">: verificato dal vivo che la scelta
# tag varia PER SQUADRA (es. Atalanta/Milan usano <strong>, Inter/Roma/
# Bologna/Como/Fiorentina/Frosinone/Sassuolo/Torino/Udinese usano <span>)
# ma la struttura interna e' identica - il pattern accetta entrambi.
# Segue ":" (opzionale) e poi il testo libero in un ulteriore
# <strong class="text-strong">.
BLOCCO_QUALITATIVO_RE = re.compile(
    r'<strong class="text-strong"><(strong|span) style="color:#[0-9a-fA-F]{6}">'
    r'<span style="font-size:150%;line-height:116%">([^<:]+):?</span>'
    r'</\1></strong>\s*<strong class="text-strong">(.*?)</strong>',
    re.DOTALL,
)

TAG_RE = re.compile(r"<[^>]+>")


def pulisci_testo(raw_html):
    """Rimuove tag HTML residui (es. smiley <img class="smilies">) e
    decodifica le entities, normalizzando gli spazi."""
    testo = TAG_RE.sub(" ", raw_html)
    testo = html_lib.unescape(testo)
    testo = re.sub(r"\s+", " ", testo).strip()
    return testo


def estrai_sezioni(html):
    """Ritorna dict {categoria: html_segmento} isolando il testo tra
    l'inizio di una sezione nota (CONSIGLIATI/POSSIBILI.SORPRESE/
    SCONSIGLIATI) e il marcatore-immagine SUCCESSIVO QUALSIASI (anche
    anonimo/senza nome) - questo garantisce che una sotto-sezione senza
    nome (es. giocatori multiruolo, vista dopo SCONSIGLIATI su Atalanta)
    non venga inclusa per errore nel segmento precedente. Se un
    marcatore noto non e' trovato la sezione risulta assente (non
    forziamo nulla)."""
    tutti_i_marker = [(m.start(), m.end(), m.group(1)) for m in MARKER_ANY_RE.finditer(html)]

    inizio_sezione_nota = {}
    for start, end, nome_marker in tutti_i_marker:
        if nome_marker in SEZIONI_ORDINE and nome_marker not in inizio_sezione_nota:
            inizio_sezione_nota[nome_marker] = end

    segmenti = {}
    for sezione in ("CONSIGLIATI", "POSSIBILI.SORPRESE", "SCONSIGLIATI"):
        if sezione not in inizio_sezione_nota:
            continue
        inizio = inizio_sezione_nota[sezione]
        # fine = inizio del prossimo marcatore-immagine QUALSIASI (con o
        # senza nome) che compare dopo "inizio"
        successivi = [start for start, _, _ in tutti_i_marker if start >= inizio]
        fine = min(successivi) if successivi else inizio + 20000
        segmenti[sezione] = html[inizio:fine]
    return segmenti


def parse_qualitativo(html):
    """Ritorna lista di dict {categoria, nome_forum, analisi_forum}."""
    risultati = []
    segmenti = estrai_sezioni(html)
    for sezione, segmento in segmenti.items():
        categoria = CATEGORIA_PER_SEZIONE[sezione]
        for m in BLOCCO_QUALITATIVO_RE.finditer(segmento):
            _tag, nome_raw, testo_raw = m.groups()
            nome = pulisci_testo(nome_raw).rstrip(":").strip()
            testo = pulisci_testo(testo_raw)
            if not nome or not testo:
                continue
            risultati.append({
                "categoria": categoria,
                "nome_forum": nome,
                "analisi_forum": testo,
            })
    return risultati


FIELDNAMES = ["squadra", "categoria", "nome_forum", "analisi_forum", "player_id"]


def main():
    rose_fanta = carica_quotazioni_2026_27()
    session = requests.Session()

    righe_out = []
    tot_estratti = 0
    tot_matchati = 0
    tot_non_matchati = 0
    non_matchati_dettaglio = []

    for squadra, topic_id in sorted(SQUADRA_TO_TOPIC_ID.items()):
        if squadra not in rose_fanta:
            continue
        url = f"https://forum.gruppoesperti.it/viewtopic.php?t={topic_id}"
        r = get_con_retry(session, url)
        time.sleep(2.5)
        if r is None:
            log.error("Impossibile scaricare topic per %s (t=%d) dopo retry", squadra, topic_id)
            continue

        blocchi = parse_qualitativo(r.text)
        log.info("%s (t=%d): %d blocchi qualitativi estratti (CONSIGLIATI/SORPRESE/SCONSIGLIATI)",
                  squadra, topic_id, len(blocchi))
        if not blocchi:
            log.warning("%s: nessun blocco qualitativo trovato - verificare formato pagina", squadra)
            continue
        tot_estratti += len(blocchi)

        rosa_fanta = rose_fanta[squadra]
        rosa_norm = [(nome_fanta, norm(nome_fanta), player_id) for nome_fanta, player_id in rosa_fanta.items()]

        for b in blocchi:
            # Pre-pulizia del nome forum prima di norm(): (1) rimuove
            # annotazioni di ruolo/nota tra parentesi che non fanno parte
            # del nome (es. "L.HENRIQUE (se listato C)" -> "L.HENRIQUE",
            # altrimenti i token spuri "se"/"listato"/"c" abbassano il
            # match_score sotto soglia); (2) sostituisce Ø/ø con O/o -
            # strip_accents() (via unicodedata NFD) NON scompone questo
            # carattere in lettera+diacritico (non e' precomposto in quel
            # modo), quindi "HØJLUND" diventava "h jlund" (spazio) invece
            # di "hojlund", bloccando il match con "Hojlund" (bug scoperto
            # dal vivo sul caso Napoli).
            nome_pulito = re.sub(r"\([^)]*\)", "", b["nome_forum"]).strip()
            nome_pulito = nome_pulito.replace("Ø", "O").replace("ø", "o")
            nome_norm = norm(nome_pulito)
            # Se il nome forum e' nella forma "INIZIALE.COGNOME" (es.
            # "L.HENRIQUE" per "Luis Henrique") anche il solo cognome
            # (ultimo token dopo lo split su norm, che rimuove il punto)
            # e' un candidato valido - match_score(fanta, "l henrique")
            # non trova nulla perche' "l" non e' un token di "luis
            # henrique", ma match_score(fanta, "henrique") si' (cognome
            # tra i token, score 2). Proviamo entrambe le varianti.
            varianti_norm = {nome_norm}
            if re.match(r"^[a-z]\s+\w+", nome_norm):
                varianti_norm.add(nome_norm.split(None, 1)[1])
            candidati = []
            for nome_fanta, nome_fanta_norm, player_id in rosa_norm:
                # match_score() NON e' simmetrico: qui il "nome forum" nel
                # blocco qualitativo e' spesso il solo COGNOME in
                # maiuscolo (es. "EDERSON", "CASTRO", "DAVIS") mentre il
                # nome fantacalcio ha un'iniziale disambiguante dopo il
                # cognome (es. "Ederson D.S.", "Castro S.") - passando il
                # cognome forum come primo argomento (tokens presi da
                # nome_fanta) si ottiene score=2 (cognome tra i token),
                # mentre nell'ordine opposto risulta solo "contains" =0.5
                # (sotto soglia in pratica coincide, ma teniamo comunque
                # il MASSIMO dei due ordini per robustezza, stesso
                # principio gia' usato in scrape_forum_esperti_2026_27.py
                # per cognome vs nome_forum completo).
                candidati_score = []
                for variante in varianti_norm:
                    candidati_score.append(match_score(nome_fanta_norm, variante))
                    candidati_score.append(match_score(variante, nome_fanta_norm))
                candidati_score = [s for s in candidati_score if s is not None]
                score = max(candidati_score) if candidati_score else None
                if score is not None and score >= MIN_MATCH_SCORE:
                    candidati.append((score, player_id))
            candidati.sort(key=lambda c: -c[0])
            player_id = candidati[0][1] if candidati else None
            if player_id is None:
                tot_non_matchati += 1
                non_matchati_dettaglio.append(f"{squadra}: {b['nome_forum']} ({b['categoria']})")
            else:
                tot_matchati += 1
            righe_out.append({
                "squadra": squadra,
                "categoria": b["categoria"],
                "nome_forum": b["nome_forum"],
                "analisi_forum": b["analisi_forum"],
                "player_id": player_id,
            })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(righe_out)

    log.info("=== Riepilogo ===")
    log.info("Blocchi qualitativi estratti (totale su 20 squadre): %d", tot_estratti)
    log.info("Match nome->player_id riusciti: %d", tot_matchati)
    log.info("Match nome->player_id FALLITI (dichiarati, non forzati): %d", tot_non_matchati)
    if non_matchati_dettaglio:
        log.warning("Dettaglio non-match:\n%s", "\n".join(non_matchati_dettaglio))
    log.info("Output scritto in %s (%d righe)", OUT_PATH, len(righe_out))


if __name__ == "__main__":
    main()
