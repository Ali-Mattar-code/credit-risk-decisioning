.PHONY: install reproduce test lint api dashboard

install:
	python -m pip install -e ".[dev,app]"

reproduce:
	credit-risk reproduce

test:
	python -m pytest

lint:
	python -m ruff check .

api:
	uvicorn credit_risk.api:app --reload

dashboard:
	streamlit run app/dashboard.py
