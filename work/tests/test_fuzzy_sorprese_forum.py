"""Test funzionali per fuzzy_sorprese_forum.py (sistema Mamdani via scikit-fuzzy)."""
import csv

import pytest

from conftest import import_fresh


@pytest.fixture
def fuzzy_module(pipeline_env):
    return import_fresh("fuzzy_sorprese_forum.py")


@pytest.mark.parametrize("pred_titolare,forum_titolarita,categoria_attesa", [
    # modello basso, forum alto -> sorpresa al rialzo (regola esplicita)
    (2.0, 9.5, "SORPRESA AL RIALZO"),
    # modello alto, forum basso -> rischio al ribasso (regola esplicita)
    (30.0, 1.0, "RISCHIO AL RIBASSO"),
    # modello e forum entrambi alti -> coerente
    (28.0, 9.0, "coerente"),
    # modello e forum entrambi bassi -> coerente (regola "neutro")
    (2.0, 1.0, "coerente"),
])
def test_categoria_sorpresa_sulle_soglie(fuzzy_module, pred_titolare, forum_titolarita, categoria_attesa):
    sim = fuzzy_module.costruisci_sistema_fuzzy()
    sim.input["pred_titolare"] = pred_titolare
    sim.input["forum_titolarita"] = forum_titolarita
    sim.compute()
    indice = float(sim.output["sorpresa"])
    categoria = fuzzy_module.categoria_da_indice(indice)
    assert categoria_attesa in categoria, (
        f"pred={pred_titolare} forum={forum_titolarita} -> indice={indice:.2f} categoria='{categoria}' "
        f"(atteso contenente '{categoria_attesa}')"
    )


@pytest.mark.parametrize("pred_titolare,forum_titolarita", [
    # NB: 35.0 esatto e' intenzionalmente escluso qui - vedi
    # test_bordo_35_esatto_soleva_keyerror_nel_sistema_fuzzy sotto, che
    # documenta perche' main() clampa a 34.99 e non 35.0.
    (0.0, 0.0), (0.0, 10.0), (34.99, 0.0), (34.99, 10.0),
    (15.0, 5.0), (7.5, 7.5), (33.2, 9.0), (5.0, 2.5),
])
def test_indice_sorpresa_sempre_in_range(fuzzy_module, pred_titolare, forum_titolarita):
    sim = fuzzy_module.costruisci_sistema_fuzzy()
    sim.input["pred_titolare"] = pred_titolare
    sim.input["forum_titolarita"] = forum_titolarita
    sim.compute()
    indice = float(sim.output["sorpresa"])
    assert -10.0 <= indice <= 10.0


def test_bordo_35_esatto_solleva_keyerror_nel_sistema_fuzzy(fuzzy_module):
    """Documenta il bug scoperto da questa suite: pred_titolare=35.0 ESATTO
    e' il vertice destro della membership function "alto" (trimf
    [16,26,35]), dove il grado di appartenenza e' 0 per costruzione
    (trimf vale 0 sui due estremi della base). A quel valore preciso TUTTE
    le membership di pred_titolare risultano 0, nessuna regola si attiva, e
    scikit-fuzzy lascia sim.output vuoto -> KeyError su
    sim.output["sorpresa"]. Per questo main() clampa pred_val a 34.99 (non
    35.0) prima di chiamare sim.compute() - vedi il commento nello script."""
    sim = fuzzy_module.costruisci_sistema_fuzzy()
    sim.input["pred_titolare"] = 35.0
    sim.input["forum_titolarita"] = 5.0
    sim.compute()
    with pytest.raises(KeyError):
        _ = sim.output["sorpresa"]


def test_input_vuoto_non_solleva_indexerror(fuzzy_module, pipeline_dirs):
    """Regressione per il fix difensivo: se previsioni_serie_a_2026_27.csv
    ha solo l'header (nessuna riga), main() non deve piu' sollevare
    IndexError su righe[0].keys() - deve loggare un warning e uscire senza
    scrivere il CSV di output."""
    in_path = fuzzy_module.IN_PATH
    with open(in_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    with open(in_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    fuzzy_module.main()  # non deve solzare IndexError

    assert not fuzzy_module.OUT_PATH.exists()
