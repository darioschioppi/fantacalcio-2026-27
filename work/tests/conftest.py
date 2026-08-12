"""
Fixture condivise per la suite di test E2E della pipeline fantacalcio.

Le fixture usano dati REALI (piccolo sottoinsieme di 18 player_id con storico
continuo su >=6 stagioni, estratti da work/data/*.csv tramite
work/tests/_extract_fixtures.py) invece di dati sintetici, per preservare
edge case genuini (cambi squadra, mancati match Understat, anagrafiche
assenti). Ogni test che esegue uno script della pipeline lo fa puntare, via
le env var FANTACALCIO_DATA_DIR / FANTACALCIO_MODELS_DIR, a una directory
temporanea popolata con queste fixture - gli script di produzione (in
work/scripts/) NON vengono mai eseguiti sui dati reali dai test, e il
comportamento di default (nessuna env var impostata) resta invariato.
"""
import importlib.util
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "work" / "scripts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# mappa: nome file reale atteso dagli script -> nome file fixture in fixtures/
FIXTURE_FILES = {
    "voti_storici_2015_2026.csv": "voti_storici_sample.csv",
    "understat_player_match_stats_storico_2015_2026.csv": "understat_sample.csv",
    "player_name_mapping.csv": "player_name_mapping_sample.csv",
    "classifica_dinamica_storico_2015_2026.csv": "classifica_dinamica_sample.csv",
    "quotazioni_fantacalcio_storico_2015_2026.csv": "quotazioni_sample.csv",
    "eta_giocatori_storico_2015_2026.csv": "eta_sample.csv",
    "infortuni_giocatori_storico_2015_2026.csv": "infortuni_sample.csv",
    "profilo_giocatori_storico_2015_2026.csv": "profilo_sample.csv",
    "forum_esperti_pagelle_2026_27.csv": "forum_esperti_sample.csv",
}


@pytest.fixture
def pipeline_dirs(tmp_path):
    """Crea data/ e models/ in una dir temporanea, popolando data/ con le
    fixture reali sotto i nomi file attesi dagli script di produzione."""
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    for real_name, fixture_name in FIXTURE_FILES.items():
        shutil.copy(FIXTURES_DIR / fixture_name, data_dir / real_name)
    # previsioni_serie_a_2026_27.csv e' l'INPUT di fuzzy_sorprese_forum.py
    # (normalmente prodotto da predict_serie_a_2026_27.py): i test che
    # usano solo fuzzy_module (senza aver rieseguito predict) partono da
    # questa fixture pre-esistente.
    previsioni_fixture = FIXTURES_DIR / "previsioni_sample.csv"
    if previsioni_fixture.exists():
        shutil.copy(previsioni_fixture, data_dir / "previsioni_serie_a_2026_27.csv")
    return {"data_dir": data_dir, "models_dir": models_dir}


@pytest.fixture
def pipeline_env(pipeline_dirs, monkeypatch):
    """Punta le env var FANTACALCIO_DATA_DIR/MODELS_DIR alla dir temporanea
    con le fixture. Usare insieme a import_fresh() per garantire che i
    moduli rilegano i path aggiornati (sono costanti calcolate a import-time)."""
    monkeypatch.setenv("FANTACALCIO_DATA_DIR", str(pipeline_dirs["data_dir"]))
    monkeypatch.setenv("FANTACALCIO_MODELS_DIR", str(pipeline_dirs["models_dir"]))
    return pipeline_dirs


def import_fresh(filename):
    """Importa (o re-importa da zero) uno script di work/scripts/ come modulo
    Python isolato, così le sue costanti module-level (DATA_DIR, MODELS_DIR,
    logging handlers) vengono ricalcolate leggendo le env var CORRENTI.
    Ogni chiamata usa un nome di modulo univoco per evitare che sys.modules
    conservi lo stato (in particolare i logging.FileHandler già apert) di
    un'esecuzione precedente con path diversi."""
    path = SCRIPTS_DIR / filename
    mod_name = f"_pipeline_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build_dataset_module(pipeline_env):
    """Esegue build_stagione_giocatore_dataset.main() sulle fixture e
    restituisce (modulo, path_output) per i test successivi."""
    module = import_fresh("build_stagione_giocatore_dataset.py")
    module.main()
    out_path = module.OUT_PATH
    assert out_path.exists(), "build_stagione_giocatore_dataset non ha scritto il CSV di output atteso"
    return module, out_path
