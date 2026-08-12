"""Test della guardia anti-leakage verifica_anti_leakage() in build_stagione_giocatore_dataset.py."""
import pytest

from conftest import import_fresh


@pytest.fixture
def build_module(pipeline_env):
    return import_fresh("build_stagione_giocatore_dataset.py")


def test_colonna_vietata_fa_uscire_con_systemexit(build_module):
    fieldnames_ok = ["stagione_target", "player_id", "fantamedia_target"]
    fieldnames_bad = fieldnames_ok + ["minuti_totali_target"]
    with pytest.raises(SystemExit) as exc_info:
        build_module.verifica_anti_leakage(fieldnames_bad)
    assert exc_info.value.code == 1


@pytest.mark.parametrize("colonna_vietata", [
    "minuti_totali_target", "xg_totale_target", "xa_totale_target",
    "shots_totali_target", "quotazione_attuale_target", "fvm_target",
    "eta_target", "infortuni_target_count", "infortuni_n_count", "altezza_target",
])
def test_ciascuna_colonna_vietata_individualmente(build_module, colonna_vietata):
    fieldnames = ["stagione_target", "player_id", colonna_vietata]
    with pytest.raises(SystemExit):
        build_module.verifica_anti_leakage(fieldnames)


def test_output_normale_non_fa_scattare_leakage(build_module):
    """Le colonne reali prodotte dal builder (incluse le eccezioni dichiarate
    quotazione_iniziale_target, squadra_in_champions_target, voto_medio_target)
    non devono mai far scattare SystemExit."""
    fieldnames_reali = [
        "stagione_target", "player_id", "nome_giocatore", "ruolo",
        "fantamedia_target", "gol_target", "assist_target", "bonus_netti_target",
        "presenze_target", "voto_medio_target", "presenze_titolare_target",
        "quotazione_iniziale_target", "squadra_in_champions_target",
    ]
    # non deve solzare SystemExit
    build_module.verifica_anti_leakage(fieldnames_reali)


def test_dataset_costruito_dalle_fixture_passa_la_verifica(build_dataset_module):
    """End-to-end: il dataset reale costruito su fixture deve gia' aver
    passato verifica_anti_leakage() dentro main() senza SystemExit (altrimenti
    build_dataset_module non sarebbe arrivato a scrivere out_path)."""
    module, out_path = build_dataset_module
    assert out_path.exists()
