.PHONY: run backfill gold gold_vehicle gold_backfill train eval test lint clean

# =========================
# Phase 1 — Silver Pipeline
# =========================

run:
	python -m scripts.run_day --dt $(dt)

backfill:
	python -m scripts.backfill --start $(start) --end $(end)

# =========================
# Phase 2 — Gold Pipeline
# =========================

gold:
	python -m scripts.run_gold_day --dt $(dt)

gold_vehicle:
	python -m scripts.run_gold_day --dt $(dt) --vehicle_id $(vehicle_id)

gold_backfill:
	python -m scripts.backfill_gold --start $(start) --end $(end)

# =========================
# Phase 3 — Training / Eval
# =========================

train:
	python -m scripts.train_soc --dt $(dt) --vehicle_id $(vehicle_id)

eval:
	python -m scripts.eval_soc --dt $(dt) --vehicle_id $(vehicle_id)

# =========================
# Testing
# =========================

test:
	pytest -q

# =========================
# Development
# =========================

lint:
	python -m pyflakes ingestion features scripts tests || true

# =========================
# Cleanup
# =========================

clean:
	find data -name ".tmp*" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# =========================
# Reports
# =========================
report:
	python scripts/model_report.py --dt=$(dt) --vehicle=$(vehicle) --model=$(model) $(shap)