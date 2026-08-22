#!/usr/bin/env python3
"""
Scraper delle "schede partita" del forum Gruppo Esperti
(forum.gruppoesperti.it), richiesto esplicitamente da Dario dopo aver
condiviso il link al topic Roma-Fiorentina: "Analizza tutti i thread
delle partite della prima giornata per aumentare i dati della
valutazione".

CONTESTO: la board "FANTACALCIO GE | Schede squadra e schede partita"
(viewforum.php?f=199) contiene, oltre ai 20 topic "SQUADRA [TOPIC
UNICO]" gia' scrapati da scrape_forum_esperti_2026_27.py, una sezione
sorella di "schede partita": un topic per singola partita, con
contenuto diverso e piu' specifico per la giornata in questione
(probabile formazione, ballottaggi %, indisponibili, voto consigliato
1/5-5/5 per ogni giocatore della rosa, rigoristi/piazzati). Individuati
dal vivo i 10 topic della giornata 1 (stagione 2026-27), tutti su f=199:

  Udinese-Como=232988, Inter-Monza=233015, Genoa-Napoli=232941,
  Parma-Cagliari=232979, Frosinone-Juventus=232945, Venezia-Lecce=232980,
  Atalanta-Sassuolo=232981, Torino-Milan=232985, Bologna-Lazio=233011,
  Roma-Fiorentina=232952.

Sigle squadra e date/ore UTC prese da calendario_data.js (giornata 1,
stagione 2026-27) - NB: la sigla Lecce nel calendario e' "USL", non
"LEC" come nel resto del progetto (SQUADRA_TO_TOPIC_ID di
scrape_forum_esperti_2026_27.py usa "LEC"): qui manteniamo "LEC" per
coerenza con quotazioni_fantacalcio_storico_2015_2026.csv (che usa LEC),
dato che il matching player_id si basa su quel file, non su
calendario_data.js.

FORMATO POST (verificato dal vivo su Roma-Fiorentina t=232952 e
Atalanta-Sassuolo t=232981, poi validato su altri 8 topic): OGNI
partita ha DUE post principali (uno per squadra, autore = nome ufficiale
squadra, es. "AS Roma"/"ACF Fiorentina"), seguiti da post di discussione
utenti (IGNORATI - solo i 2 post ufficiali per squadra contengono dati
strutturati). Ogni post ufficiale contiene, in ordine, delimitato da
marcatori-immagine <img src=".../HASH.png[/NOME]">:

  PROBABILE.FORMAZIONE -> modulo + probabile XI (nomi in grassetto,
      alcuni con <strong style="color:#bf0000"> per segnalare un
      giocatore in ballottaggio incluso comunque nell'undici)
  BALLOTTAGGI          -> blocchi "Nome [XX%] - Nome2 [YY%] (- Nome3 [ZZ%])"
      con eventuale nota libera sotto (non sempre presente)
  (marcatore anonimo, "possibilita' di voto" qualitativa Bassa/Media/Alta
      per l'intera rosa - non estratto: gia' ridondante col voto 1-5/5
      dedicato piu' sotto, e usa etichette libere non numeriche)
  (2 marcatori anonimi, diffidati/squalificati - spesso "-"/"Nessuno",
      non estratti: non richiesti dal task, valore raro/poco strutturato)
  (marcatore anonimo)  -> INDISPONIBILI: blocco libero "Nome - motivo -
      rientro previsto" (talvolta piu' nomi su piu' righe, talvolta
      "Nome1, Nome2 - motivo unico")
  RIGORISTI             -> "- Nome1, Nome2, ..."
  CALCI.PIAZZATI        -> "- Punizioni: ..." e "- Angoli: ..." su righe
      separate
  (marcatore anonimo)   -> LEGENDA (mappa colore->1/5..5/5, non estratta,
      serve solo a noi umani per leggere la sezione voto)
  PORTIERI / DIFENSORI / CENTROCAMPISTI / ATTACCANTI -> blocco voto:
      per OGNI giocatore della rosa (non solo titolari) un blocco
      "<strong><span style='color:#HEX'>X/5</span> Nome</strong> -
      commento libero (a volte assente/solo rimando a CONSIGLIATI)"
  CONSIGLIATI / POSSIBILI.SORPRESE / SCONSIGLIATI -> approfondimento
      testuale libero (STESSO formato di
      scrape_forum_esperti_qualitativo_2026_27.py, "- NOME: testo") -
      NON ri-estratto qui: e' arricchimento del voto 1/5-5/5 gia'
      catturato, duplicarlo in un'altra colonna non aggiunge segnale
      nuovo per l'indice di rischio richiesto.
  COPYRIGHT             -> fine post, terminatore.

Verificato dal vivo che l'attributo colore del voto puo' essere sia
<strong style="color:#00BF00"> (hex) sia <strong style="color:red">
(nome CSS) - entrambi accettati dalla regex del voto.

MATCHING NOME -> player_id: riuso di norm/match_score/MIN_MATCH_SCORE/
carica_quotazioni_2026_27 da scrape_forum_esperti_2026_27.py (import
diretto, stesso pattern di scrape_forum_esperti_qualitativo_2026_27.py),
ristretto alla SOLA squadra del post corrente (piu' affidabile che su
tutta la Serie A). Non-match dichiarati e loggati esplicitamente, mai
forzati - stesso principio anti-invenzione-dati del resto del progetto.

Output:
  work/data/schede_partita_giornata1_2026_27.csv
    giornata, squadra_casa, squadra_ospite, data_ora, squadra, player_id,
    nome_forum, titolare_previsto, ballottaggio_pct, indisponibile,
    motivo_indisponibilita, voto_consigliato, commento_voto, rigorista,
    tira_punizioni, tira_angoli
  work/data/scrape_schede_partita_log.txt

Uso:
  python3 scrape_schede_partita_2026_27.py
"""
import csv
import html as html_lib
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from scrape_forum_esperti_2026_27 import (
    HEADERS,
    carica_quotazioni_2026_27,
    get_con_retry,
    match_score,
    norm,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "schede_partita_giornata1_2026_27.csv"
LOG_PATH = DATA_DIR / "scrape_schede_partita_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("scrape_schede_partita")

MIN_MATCH_SCORE = 1

# (topic_id) -> (squadra_casa, squadra_ospite, data_ora_utc) - sigle
# coerenti con quotazioni_fantacalcio_storico_2015_2026.csv (LEC, non
# USL come in calendario_data.js), date/ore da calendario_data.js
# giornata 1 stagione 2026-27.
TOPIC_TO_PARTITA = {
    232988: ("UDI", "COM", "2026-08-22T16:30:00Z"),
    233015: ("INT", "MON", "2026-08-22T16:30:00Z"),
    232941: ("GEN", "NAP", "2026-08-22T18:45:00Z"),
    232979: ("PAR", "CAG", "2026-08-22T18:45:00Z"),
    232945: ("FRO", "JUV", "2026-08-23T16:30:00Z"),
    232980: ("VEN", "LEC", "2026-08-23T16:30:00Z"),
    232981: ("ATA", "SAS", "2026-08-23T18:45:00Z"),
    232985: ("TOR", "MIL", "2026-08-23T18:45:00Z"),
    233011: ("BOL", "LAZ", "2026-08-24T16:30:00Z"),
    232952: ("ROM", "FIO", "2026-08-24T18:45:00Z"),
}
GIORNATA = 1

# Marcatore-immagine con eventuale suffisso testuale nell'URL (es.
# ".../XHASH.png/PROBABILE.FORMAZIONE"); alcuni marcatori sono anonimi
# (nessun suffisso) - stesso pattern di riconoscimento gia' usato in
# scrape_forum_esperti_qualitativo_2026_27.py (MARKER_ANY_RE).
MARKER_RE = re.compile(
    r'<img[^>]*src="([^"]*\.(?:png|jpe?g|gif))"[^>]*>',
)

# Blocco voto 1/5-5/5: colore sia hex (#00BF00) sia nome CSS (red),
# entrambi verificati dal vivo (Roma/Fiorentina usano hex, Atalanta usa
# sia hex che "red" per il 2/5 - incoerenza del forum, gestita). Il
# wrapper esterno <strong class="text-strong"> attorno al tag colorato e'
# OPZIONALE - verificato dal vivo sul post Sassuolo (t=232981, 2o post),
# che usa direttamente <strong style="color:red">2/5 Muric</strong> SENZA
# wrapper (mentre Roma/Atalanta-casa lo hanno sempre) - senza questo fix
# il post Sassuolo produceva 0 voti estratti nonostante 20 blocchi
# presenti nell'HTML.
VOTO_RE = re.compile(
    r'(?:<strong class="text-strong">\s*)?<(?:strong|span) style="color:(?:#[0-9a-fA-F]{6}|[a-zA-Z]+)">'
    r'\s*([1-5])/5\s*(?:</(?:strong|span)>)?\s*([^<]*?)\s*</strong>'
)

TAG_RE = re.compile(r"<[^>]+>")
PCT_RE = re.compile(r"([^\[\],]+?)\s*\[(\d+)\s*%\]")


def pulisci_testo(raw_html):
    testo = TAG_RE.sub(" ", raw_html)
    testo = html_lib.unescape(testo)
    testo = re.sub(r"\s+", " ", testo).strip()
    return testo


def trova_marcatori(post_html):
    """Ritorna lista di (start, end, nome_o_None) per ogni marcatore-
    immagine nel post, in ordine di apparizione. Il nome e' l'ultimo
    segmento del path URL se contiene un '.', altrimenti None (marcatore
    anonimo) - le immagini smilies (es. ./images/smilies/pippotto.gif)
    NON hanno mai un secondo '/' dopo il dominio quindi non verrebbero
    scambiate per marcatori di sezione, ma per sicurezza le riconosciamo
    solo tra quelle con host imgur.com."""
    marcatori = []
    for m in re.finditer(r'<img[^>]*src="([^"]+)"[^>]*>', post_html):
        src = m.group(1)
        if "imgur.com" not in src:
            continue
        parts = src.split(".com/", 1)[-1]
        nome = None
        if "/" in parts.split(".", 1)[-1] or parts.count("/") >= 1:
            segs = parts.split("/")
            if len(segs) > 1 and segs[-1]:
                nome = segs[-1].upper()
        marcatori.append((m.start(), m.end(), nome))
    return marcatori


def segmento_dopo(post_html, marcatori, nome_target):
    """Ritorna l'HTML tra la fine del PRIMO marcatore con nome esatto
    nome_target e l'inizio del marcatore imgur SUCCESSIVO (con o senza
    nome) - stesso principio di robustezza di estrai_sezioni() in
    scrape_forum_esperti_qualitativo_2026_27.py: fermarsi al prossimo
    marcatore QUALSIASI evita di inglobare la sezione dopo per errore."""
    inizio = None
    for start, end, nome in marcatori:
        if nome == nome_target:
            inizio = end
            break
    if inizio is None:
        return None
    successivi = [start for start, _, _ in marcatori if start >= inizio]
    fine = min(successivi) if successivi else len(post_html)
    return post_html[inizio:fine]


def parse_ballottaggi(segmento):
    """Ritorna dict {nome_norm_originale: pct} da blocchi
    "Nome [XX%] - Nome2 [YY%]" dentro <strong>. Le note libere sotto
    (senza '%') sono ignorate. Nomi conservati come appaiono (pulizia
    minima) per il matching successivo."""
    risultato = {}
    for sm in re.finditer(r'<strong class="text-strong">(.*?)</strong>', segmento, re.DOTALL):
        blocco = sm.group(1)
        if "%" not in blocco:
            continue
        blocco_clean = pulisci_testo(blocco)
        for nome_raw, pct in PCT_RE.findall(blocco_clean):
            nome = nome_raw.strip().lstrip("-").strip()
            if nome:
                risultato[nome] = int(pct)
    return risultato


def parse_indisponibili(segmento):
    """Ritorna lista di (nome, motivo) da blocchi liberi
    "Nome - motivo - rientro" o "Nome1, Nome2 - motivo unico"."""
    risultato = []
    testo = pulisci_testo(segmento)
    if not testo or testo.strip("- ").strip() == "" or testo.strip().lower() in ("nessuno", "-"):
        return risultato
    # split su newline non disponibile (testo gia' appiattito) - usiamo
    # invece i blocchi <strong>Nome</strong> del segmento originale come
    # ancora, poi prendiamo il testo fino al prossimo <strong>.
    blocchi = list(re.finditer(r'<strong class="text-strong">([^<]+)</strong>', segmento))
    for i, bm in enumerate(blocchi):
        nome_raw = bm.group(1).strip().strip(":").strip()
        if not nome_raw or nome_raw == "-":
            continue
        fine = blocchi[i + 1].start() if i + 1 < len(blocchi) else len(segmento)
        motivo_html = segmento[bm.end():fine]
        motivo = pulisci_testo(motivo_html).lstrip("-").strip()
        # nomi multipli separati da virgola nello stesso blocco (es.
        # "Duncan, Berardi, Casas... - motivo unico") -> stesso motivo
        # per ognuno.
        for nome in nome_raw.split(","):
            nome = nome.strip()
            if nome:
                risultato.append((nome, motivo))
    return risultato


def parse_voti(segmento):
    """Ritorna lista di (voto_1_5, nome, commento) da blocchi
    "<strong>X/5</strong> Nome</strong> - commento"."""
    risultato = []
    matches = list(VOTO_RE.finditer(segmento))
    for i, m in enumerate(matches):
        voto = int(m.group(1))
        nome = m.group(2).strip()
        inizio_commento = m.end()
        fine_commento = matches[i + 1].start() if i + 1 < len(matches) else len(segmento)
        commento_html = segmento[inizio_commento:fine_commento]
        commento = pulisci_testo(commento_html).lstrip("-").strip()
        if nome:
            risultato.append((voto, nome, commento))
    return risultato


def parse_nomi_lista(testo_html):
    """Ritorna lista di nomi puliti da un blocco "- Nome1, Nome2, Nome3"
    o "Punizioni: Nome1, Nome2" (rimuove il prefisso etichetta se
    presente)."""
    testo = pulisci_testo(testo_html)
    testo = re.sub(r"^[-\s]*(?:Punizioni|Angoli)\s*:\s*", "", testo, flags=re.IGNORECASE)
    testo = testo.lstrip("-").strip()
    if not testo or testo.lower() in ("nessuno", "-"):
        return []
    return [n.strip() for n in testo.split(",") if n.strip()]


def parse_post_squadra(post_html):
    """Estrae tutti i dati strutturati di un post-squadra. Ritorna dict
    con: ballottaggi {nome: pct}, indisponibili [(nome, motivo)],
    voti [(voto, nome, commento)], rigoristi [nomi], punizioni [nomi],
    angoli [nomi]."""
    marcatori = trova_marcatori(post_html)

    seg_ballottaggi = segmento_dopo(post_html, marcatori, "BALLOTTAGGI") or ""
    ballottaggi = parse_ballottaggi(seg_ballottaggi)

    # L'INDISPONIBILI e' il marcatore anonimo subito prima di RIGORISTI:
    # troviamo l'indice del marcatore RIGORISTI e usiamo il marcatore
    # immediatamente precedente come inizio segmento.
    idx_rigoristi = next((i for i, (_, _, n) in enumerate(marcatori) if n == "RIGORISTI"), None)
    indisponibili = []
    if idx_rigoristi is not None and idx_rigoristi > 0:
        start_indisp = marcatori[idx_rigoristi - 1][1]
        end_indisp = marcatori[idx_rigoristi][0]
        indisponibili = parse_indisponibili(post_html[start_indisp:end_indisp])

    seg_rigoristi = segmento_dopo(post_html, marcatori, "RIGORISTI") or ""
    rigoristi = parse_nomi_lista(seg_rigoristi)

    seg_piazzati = segmento_dopo(post_html, marcatori, "CALCI.PIAZZATI") or segmento_dopo(post_html, marcatori, "CALCI%20PIAZZATI") or ""
    punizioni, angoli = [], []
    m_pun = re.search(r"Punizioni\s*:\s*([^<]+)", seg_piazzati, re.IGNORECASE)
    if m_pun:
        punizioni = parse_nomi_lista("Punizioni: " + m_pun.group(1))
    m_ang = re.search(r"Angoli\s*:\s*([^<]+)", seg_piazzati, re.IGNORECASE)
    if m_ang:
        angoli = parse_nomi_lista("Angoli: " + m_ang.group(1))

    # Voti: dal marcatore PORTIERI fino al PRIMO tra
    # CONSIGLIATI/POSSIBILI.SORPRESE/SCONSIGLIATI/COPYRIGHT (attraversa
    # PORTIERI/DIFENSORI/CENTROCAMPISTI/ATTACCANTI, che condividono lo
    # stesso formato di blocco voto - non serve segmentarli separatamente
    # perche' il ruolo testuale non e' richiesto in output). FERMARSI a
    # CONSIGLIATI (non proseguire fino a COPYRIGHT) e' essenziale: quella
    # sezione ha un formato diverso ("- NOME: testo libero", vedi
    # scrape_forum_esperti_qualitativo_2026_27.py) che VOTO_RE non
    # riconosce, quindi senza questo limite il commento dell'ULTIMO
    # giocatore della sezione voto (nessun prossimo match VOTO_RE entro
    # ATTACCANTI) si estendeva erroneamente fino a includere tutto il
    # testo di CONSIGLIATI/SORPRESE/SCONSIGLIATI (bug scoperto dal vivo
    # sul caso Leao/Milan: il commento_voto conteneva anche i paragrafi
    # su RAMOS G./CHUKWUEZE/CISSE'/DE WINTER).
    idx_portieri = next((i for i, (_, _, n) in enumerate(marcatori) if n == "PORTIERI"), None)
    idx_fine_voti = next(
        (i for i, (_, _, n) in enumerate(marcatori)
         if n in ("CONSIGLIATI", "POSSIBILI.SORPRESE", "SCONSIGLIATI", "COPYRIGHT")),
        None,
    )
    voti = []
    if idx_portieri is not None:
        start_voti = marcatori[idx_portieri][1]
        end_voti = marcatori[idx_fine_voti][0] if idx_fine_voti is not None else len(post_html)
        voti = parse_voti(post_html[start_voti:end_voti])

    return {
        "ballottaggi": ballottaggi,
        "indisponibili": indisponibili,
        "voti": voti,
        "rigoristi": rigoristi,
        "punizioni": punizioni,
        "angoli": angoli,
    }


def trova_nome_squadra_ufficiale(nome_autore, rosa_norm_lookup):
    """Non serve al matching (che avviene per nome giocatore, non
    squadra) - placeholder rimosso, la selezione dei 2 post ufficiali
    avviene per posizione (i primi 2 post del topic)."""
    return nome_autore


def matcha_player_id(nome_grezzo, rosa_norm):
    """Stessa strategia di scrape_forum_esperti_2026_27.py: normalizza,
    prova sia il nome completo che (se presente) il solo ultimo token
    come cognome, prende il MASSIMO punteggio. Ritorna (player_id, score)
    o (None, None) se nessun candidato sopra soglia."""
    nome_pulito = re.sub(r"\([^)]*\)", "", nome_grezzo).strip()
    nome_norm = norm(nome_pulito)
    if not nome_norm:
        return None, None
    tokens = nome_norm.split()
    varianti = {nome_norm}
    if tokens:
        varianti.add(tokens[-1])
    candidati = []
    for nome_fanta, nome_fanta_norm, player_id in rosa_norm:
        for variante in varianti:
            for score in (match_score(nome_fanta_norm, variante), match_score(variante, nome_fanta_norm)):
                if score is not None:
                    candidati.append((score, player_id))
    if not candidati:
        return None, None
    candidati.sort(key=lambda c: -c[0])
    score, player_id = candidati[0]
    if score < MIN_MATCH_SCORE:
        return None, None
    return player_id, score


FIELDNAMES = [
    "giornata", "squadra_casa", "squadra_ospite", "data_ora", "squadra",
    "player_id", "nome_forum", "titolare_previsto", "ballottaggio_pct",
    "indisponibile", "motivo_indisponibilita", "voto_consigliato",
    "commento_voto", "rigorista", "tira_punizioni", "tira_angoli",
]


def costruisci_righe_squadra(dati, squadra, rosa_fanta):
    """Combina voti/ballottaggi/indisponibili/rigoristi/piazzati in righe
    per-giocatore, matchando ogni nome sulla rosa 2026-27 della squadra.
    Una riga per ogni giocatore comparso in ALMENO una delle sezioni
    (tipicamente i voti coprono l'intera rosa, quindi guidano l'elenco)."""
    rosa_norm = [(nome, norm(nome), pid) for nome, pid in rosa_fanta.items()]

    ballottaggi_norm = {norm(n): (n, pct) for n, pct in dati["ballottaggi"].items()}
    indisponibili_norm = {norm(n): (n, motivo) for n, motivo in dati["indisponibili"]}
    rigoristi_norm = {norm(n) for n in dati["rigoristi"]}
    punizioni_norm = {norm(n) for n in dati["punizioni"]}
    angoli_norm = {norm(n) for n in dati["angoli"]}

    righe = []
    non_matchati = []
    nomi_gia_visti = set()

    for voto, nome_forum, commento in dati["voti"]:
        nome_gia = norm(nome_forum)
        if nome_gia in nomi_gia_visti:
            continue
        nomi_gia_visti.add(nome_gia)

        player_id, score = matcha_player_id(nome_forum, rosa_norm)
        if player_id is None:
            non_matchati.append(nome_forum)

        # ballottaggio: match esatto sul nome normalizzato (i nomi nei
        # ballottaggi e nei voti provengono dallo stesso post, stessa
        # forma abbreviata - piu' affidabile di un altro match_score)
        pct = None
        for bn, (_orig, bpct) in ballottaggi_norm.items():
            if bn == nome_gia or bn in nome_gia or nome_gia in bn:
                pct = bpct
                break

        indisponibile = False
        motivo = ""
        for iname, (_orig, imotivo) in indisponibili_norm.items():
            if iname == nome_gia or iname in nome_gia or nome_gia in iname:
                indisponibile = True
                motivo = imotivo
                break

        rigorista = any(rn == nome_gia or rn in nome_gia or nome_gia in rn for rn in rigoristi_norm)
        tira_punizioni = any(pn == nome_gia or pn in nome_gia or nome_gia in pn for pn in punizioni_norm)
        tira_angoli = any(an == nome_gia or an in nome_gia or nome_gia in an for an in angoli_norm)

        # titolare_previsto: True se il voto e' presente E il giocatore
        # non e' esplicitamente in ballottaggio con probabilita' <50%
        # (se in ballottaggio ma >=50% lo consideriamo comunque probabile
        # titolare, coerente con la formulazione del forum stesso, es.
        # "Wesley [80%]" viene descritto come "altissime probabilita' di
        # vederlo titolare"). Senza dato di ballottaggio, non possiamo
        # dedurre la titolarita' dal solo voto (la sezione voto copre
        # TUTTA la rosa, riserve incluse) - lasciamo None (non forzato).
        if pct is not None:
            titolare_previsto = pct >= 50
        else:
            titolare_previsto = None

        righe.append({
            "squadra": squadra,
            "player_id": player_id,
            "nome_forum": nome_forum,
            "titolare_previsto": titolare_previsto,
            "ballottaggio_pct": pct,
            "indisponibile": indisponibile,
            "motivo_indisponibilita": motivo,
            "voto_consigliato": voto,
            "commento_voto": commento,
            "rigorista": rigorista,
            "tira_punizioni": tira_punizioni,
            "tira_angoli": tira_angoli,
        })

    return righe, non_matchati


def main():
    rose_fanta = carica_quotazioni_2026_27()
    session = requests.Session()

    righe_out = []
    tot_giocatori = 0
    tot_matchati = 0
    tot_non_matchati = 0
    non_matchati_dettaglio = []

    for topic_id, (squadra_casa, squadra_ospite, data_ora) in sorted(TOPIC_TO_PARTITA.items()):
        url = f"https://forum.gruppoesperti.it/viewtopic.php?t={topic_id}"
        r = get_con_retry(session, url)
        time.sleep(2.5)
        if r is None:
            log.error("Impossibile scaricare topic %s-%s (t=%d) dopo retry", squadra_casa, squadra_ospite, topic_id)
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        contenuti = soup.select(".content")
        if len(contenuti) < 2:
            log.error("%s-%s (t=%d): meno di 2 post trovati (%d) - salto", squadra_casa, squadra_ospite, topic_id, len(contenuti))
            continue

        for squadra, post in ((squadra_casa, contenuti[0]), (squadra_ospite, contenuti[1])):
            if squadra not in rose_fanta:
                log.warning("%s: nessuna rosa 2026-27 nota, salto la scheda", squadra)
                continue
            post_html = str(post)
            dati = parse_post_squadra(post_html)
            log.info(
                "%s-%s / %s (t=%d): %d voti, %d ballottaggi, %d indisponibili, %d rigoristi",
                squadra_casa, squadra_ospite, squadra, topic_id,
                len(dati["voti"]), len(dati["ballottaggi"]), len(dati["indisponibili"]), len(dati["rigoristi"]),
            )
            if not dati["voti"]:
                log.warning("%s (t=%d): nessun blocco voto trovato - verificare formato pagina", squadra, topic_id)
                continue

            righe, non_matchati = costruisci_righe_squadra(dati, squadra, rose_fanta[squadra])
            tot_giocatori += len(righe)
            for riga in righe:
                riga["giornata"] = GIORNATA
                riga["squadra_casa"] = squadra_casa
                riga["squadra_ospite"] = squadra_ospite
                riga["data_ora"] = data_ora
                if riga["player_id"] is None:
                    tot_non_matchati += 1
                else:
                    tot_matchati += 1
                righe_out.append(riga)
            for nome in non_matchati:
                non_matchati_dettaglio.append(f"{squadra}: {nome}")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(righe_out)

    log.info("=== Riepilogo ===")
    log.info("Partite processate: %d/10", len({(r['squadra_casa'], r['squadra_ospite']) for r in righe_out}))
    log.info("Giocatori estratti (voti, totale su 20 squadre): %d", tot_giocatori)
    log.info("Match nome->player_id riusciti: %d", tot_matchati)
    log.info("Match nome->player_id FALLITI (dichiarati, non forzati): %d", tot_non_matchati)
    if non_matchati_dettaglio:
        log.warning("Dettaglio non-match:\n%s", "\n".join(non_matchati_dettaglio))
    log.info("Output scritto in %s (%d righe)", OUT_PATH, len(righe_out))


if __name__ == "__main__":
    main()
