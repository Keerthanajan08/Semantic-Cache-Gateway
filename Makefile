.PHONY: install test test-all lint format typecheck check run demo eval docker-up docker-down

PIP_EXTRA = --extra-index-url https://download.pytorch.org/whl/cpu

install:        ## install runtime + dev dependencies (CPU torch)
	pip install $(PIP_EXTRA) -r requirements-dev.txt

test:           ## fast suite with coverage (no model download)
	pytest --cov --cov-report=term-missing

test-all:       ## fast + real-model integration tests
	pytest -m ""

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

typecheck:
	mypy

check: lint typecheck test   ## everything CI runs

run:
	uvicorn app.main:app --reload

demo:           ## fire paraphrased queries at a running server
	python scripts/run_demo.py

eval:           ## threshold sweep -> eval/results.md + eval/threshold_sweep.png
	python scripts/evaluate_threshold.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
