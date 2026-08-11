#!/usr/bin/env python3
"""
Sistema di inferenza FUZZY (Mamdani, richiesto da Dario: "Puoi usare la
logica fuzzy") per integrare il giudizio QUALITATIVO del forum Gruppo
Esperti con le previsioni QUANTITATIVE del modello LightGBM, individuando
"sorprese" (giocatori sottovalutati o sopravvalutati dal modello rispetto
al giudizio degli esperti).

PERCHE' FUZZY E NON UNA FEATURE DIRETTA NEL MODELLO: verificato che le 6
colonne forum_* (titolarita_forum, media_voto_forum, salute_forum,
bonus_forum, consiglio_esperti_forum, totale_forum), aggiunte come
feature dirette in build_stagione_giocatore_dataset.py, hanno IMPORTANZA
ESATTAMENTE 0.0 nei modelli LightGBM per tutti i 7 target - il motivo e'
strutturale: queste colonne esistono SOLO per la stagione 2026-27, che
non fa parte del train/val/test set (mai giocata), quindi sono sempre
NaN durante l'addestramento e LightGBM non ha mai potuto imparare alcuna
relazione. Un sistema fuzzy con regole esperte scritte a mano, applicato
in POST-PROCESSING sulle previsioni gia' generate, bypassa questo limite
strutturale: non serve training, le regole sono dichiarate direttamente.

INPUT (2 variabili fuzzy, entrambe gia' presenti in
previsioni_serie_a_2026_27.csv da predict_serie_a_2026_27.py):
  1. pred_presenze_titolare_previste (0-38, quanto il MODELLO prevede
     partite da titolare, basato su storico oggettivo)
  2. titolarita_forum_esperti (0-10, quanto gli ESPERTI del forum
     giudicano il giocatore titolare per la stagione 2026-27, giudizio
     qualitativo/preseason - assente per definizione dal training)

OUTPUT (1 variabile fuzzy):
  indice_sorpresa (-10..+10): positivo = "sorpresa al rialzo" (gli
  esperti vedono piu' titolarita' di quanta il modello, basato solo sul
  passato, preveda - es. un giocatore emergente/nuovo acquisto su cui il
  forum ha informazioni preseason che il modello non puo' avere) -
  negativo = "rischio al ribasso" (il modello prevede piu' titolarita' di
  quanta gli esperti si aspettino - es. un titolare storico ora in
  ballottaggio per un nuovo arrivo, informazione preseason che il
  modello, guardando solo lo storico, non vede).

Righe SENZA giudizio forum (non matchato dallo scraper, o giocatore
senza pagella assegnata sul forum, "x/10") vengono ESCLUSE dal calcolo
(indice_sorpresa=None), non forzate a un valore neutro fittizio - stesso
principio anti-invenzione-dati applicato ovunque nel progetto.

Uso:
  python3 fuzzy_sorprese_forum.py
Output:
  work/data/previsioni_serie_a_2026_27_con_sorprese.csv (tutte le colonne
    di previsioni_serie_a_2026_27.csv + indice_sorpresa + categoria)
  work/data/fuzzy_sorprese_log.txt
"""
import csv
import logging
from pathlib import Path

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH = DATA_DIR / "previsioni_serie_a_2026_27.csv"
OUT_PATH = DATA_DIR / "previsioni_serie_a_2026_27_con_sorprese.csv"
LOG_PATH = DATA_DIR / "fuzzy_sorprese_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("fuzzy_sorprese_forum")


def costruisci_sistema_fuzzy():
    """Sistema Mamdani a 2 input / 1 output, costruito con scikit-fuzzy.

    CALIBRAZIONE (11/08, corretta dopo verifica numerica): le membership
    function iniziali erano tarate sul range TEORICO 0-38 presenze,
    ignorando la distribuzione REALE delle previsioni 2026-27 (verificato
    su previsioni_serie_a_2026_27.csv: mediana=15.0, 90 percentile=25.1,
    massimo=33.2 - nessun giocatore, nemmeno il piu' titolare, si
    avvicina a 38 perche' il calendario Serie A ha 38 giornate ma nessuno
    gioca tutte le partite da titolare al 100%). Con la vecchia
    calibrazione, giocatori CHIARAMENTE titolari sia per il modello
    (~22-26 presenze previste) sia per gli esperti (voto 8-9) venivano
    classificati "alto" solo parzialmente (es. pred=24.3 -> alto=0.14),
    facendo scattare erroneamente regole di "sorpresa" su casi coerenti
    (es. Barella, Cambiaso, Di Lorenzo - tutti titolari indiscutibili).
    Ricalibrato sui percentili reali (10%=6.5, 50%=15, 75%=20.4, 90%=25.1,
    max=33.2) cosi' che "alto" catturi davvero la fascia dei titolari
    fissi e "sorpresa"/"rischio" scattino solo su vere divergenze.
    """
    # Input 1: previsione modello (0-35 presenze da titolare previste,
    # range REALE osservato ~0-33.2, esteso leggermente a 35 per evitare
    # un edge-case di floating point/aggregazione degenere quando un
    # valore clampato coincide esattamente col vertice della membership
    # function "alto" - verificato dal vivo: causava un KeyError in
    # defuzzificazione per il valore massimo esatto osservato, 33.2)
    pred_titolare = ctrl.Antecedent(np.arange(0, 35.01, 0.1), "pred_titolare")
    pred_titolare["basso"] = fuzz.trimf(pred_titolare.universe, [0, 0, 10])
    pred_titolare["medio"] = fuzz.trimf(pred_titolare.universe, [5, 14, 23])
    pred_titolare["alto"] = fuzz.trimf(pred_titolare.universe, [16, 26, 35])

    # Input 2: giudizio esperti forum (0-10)
    forum_titolarita = ctrl.Antecedent(np.arange(0, 10.01, 0.1), "forum_titolarita")
    forum_titolarita["basso"] = fuzz.trimf(forum_titolarita.universe, [0, 0, 4])
    forum_titolarita["medio"] = fuzz.trimf(forum_titolarita.universe, [2, 5, 8])
    forum_titolarita["alto"] = fuzz.trimf(forum_titolarita.universe, [6, 10, 10])

    # Output: indice sorpresa (-10 rischio ribasso .. +10 sorpresa rialzo)
    sorpresa = ctrl.Consequent(np.arange(-10, 10.01, 0.1), "sorpresa")
    sorpresa["ribasso_forte"] = fuzz.trimf(sorpresa.universe, [-10, -10, -4])
    sorpresa["neutro"] = fuzz.trimf(sorpresa.universe, [-5, 0, 5])
    sorpresa["rialzo_forte"] = fuzz.trimf(sorpresa.universe, [4, 10, 10])

    # Regole esperte (9 combinazioni, dichiarate a mano - questo e' il
    # punto centrale della logica fuzzy richiesta: nessun training, solo
    # regole di buon senso su come la DIVERGENZA tra le due fonti genera
    # un segnale di sorpresa).
    regole = [
        # modello basso, forum alto -> forte sorpresa al rialzo (il forum
        # sa qualcosa preseason che il modello, guardando solo il
        # passato, non puo' sapere - es. nuovo titolare designato)
        ctrl.Rule(pred_titolare["basso"] & forum_titolarita["alto"], sorpresa["rialzo_forte"]),
        ctrl.Rule(pred_titolare["basso"] & forum_titolarita["medio"], sorpresa["neutro"]),
        ctrl.Rule(pred_titolare["basso"] & forum_titolarita["basso"], sorpresa["neutro"]),
        ctrl.Rule(pred_titolare["medio"] & forum_titolarita["alto"], sorpresa["rialzo_forte"]),
        ctrl.Rule(pred_titolare["medio"] & forum_titolarita["medio"], sorpresa["neutro"]),
        ctrl.Rule(pred_titolare["medio"] & forum_titolarita["basso"], sorpresa["ribasso_forte"]),
        # modello alto, forum basso -> forte rischio al ribasso (era
        # titolare "sulla carta" storica, ma gli esperti segnalano un
        # ballottaggio/rischio che il modello non puo' vedere)
        ctrl.Rule(pred_titolare["alto"] & forum_titolarita["alto"], sorpresa["neutro"]),
        ctrl.Rule(pred_titolare["alto"] & forum_titolarita["medio"], sorpresa["ribasso_forte"]),
        ctrl.Rule(pred_titolare["alto"] & forum_titolarita["basso"], sorpresa["ribasso_forte"]),
    ]

    sistema = ctrl.ControlSystem(regole)
    return ctrl.ControlSystemSimulation(sistema)


def categoria_da_indice(indice):
    if indice is None:
        return None
    if indice >= 3.0:
        return "SORPRESA AL RIALZO (esperti piu' fiduciosi del modello)"
    if indice <= -3.0:
        return "RISCHIO AL RIBASSO (esperti meno fiduciosi del modello)"
    return "coerente (modello ed esperti d'accordo)"


def main():
    if not IN_PATH.exists():
        log.error("File previsioni non trovato (%s) - esegui prima predict_serie_a_2026_27.py", IN_PATH)
        raise SystemExit(1)

    sim = costruisci_sistema_fuzzy()

    with open(IN_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        righe = list(reader)

    n_con_forum = 0
    n_senza_forum = 0
    n_sorpresa_rialzo = 0
    n_rischio_ribasso = 0
    dettaglio_sorprese = []

    for r in righe:
        titolarita_forum_raw = r.get("titolarita_forum_esperti")
        pred_titolare_raw = r.get("pred_presenze_titolare_previste")
        if not titolarita_forum_raw or titolarita_forum_raw == "":
            r["indice_sorpresa"] = ""
            r["categoria_sorpresa"] = ""
            n_senza_forum += 1
            continue

        forum_val = float(titolarita_forum_raw)
        pred_val = min(max(float(pred_titolare_raw), 0.0), 35.0)

        sim.input["pred_titolare"] = pred_val
        sim.input["forum_titolarita"] = forum_val
        sim.compute()
        indice = float(sim.output["sorpresa"])

        r["indice_sorpresa"] = round(indice, 2)
        r["categoria_sorpresa"] = categoria_da_indice(indice)
        n_con_forum += 1

        if indice >= 3.0:
            n_sorpresa_rialzo += 1
            dettaglio_sorprese.append((indice, r["squadra"], r["nome"], pred_val, forum_val, "RIALZO"))
        elif indice <= -3.0:
            n_rischio_ribasso += 1
            dettaglio_sorprese.append((indice, r["squadra"], r["nome"], pred_val, forum_val, "RIBASSO"))

    fieldnames = list(righe[0].keys())
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(righe)

    log.info("=== Sistema fuzzy sorprese: %d giocatori valutati, %d esclusi (nessun giudizio forum) ===", n_con_forum, n_senza_forum)
    log.info("Sorprese al RIALZO (indice>=3): %d", n_sorpresa_rialzo)
    log.info("Rischi al RIBASSO (indice<=-3): %d", n_rischio_ribasso)
    log.info("Output scritto in %s", OUT_PATH)
    log.info("")
    log.info("=== Dettaglio sorprese/rischi (ordinate per intensita' indice) ===")
    dettaglio_sorprese.sort(key=lambda x: -abs(x[0]))
    for indice, squadra, nome, pred_val, forum_val, tipo in dettaglio_sorprese:
        log.info("  [%s] %-6s %-25s pred_titolare=%.1f forum_titolarita=%.0f indice=%.2f",
                  tipo, squadra, nome, pred_val, forum_val, indice)


if __name__ == "__main__":
    main()
