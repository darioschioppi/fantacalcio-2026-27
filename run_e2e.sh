#!/usr/bin/env bash
# Ciclo rapido di test funzionali E2E per la pipeline ML fantacalcio-2026-27.
# Esclude i test @pytest.mark.slow (che allenano sul dataset REALE intero
# per confrontare le metriche con i valori di riferimento) - quelli vanno
# lanciati manualmente con: pytest work/tests -m slow -q
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pytest work/tests -m "not slow" -q "$@"
