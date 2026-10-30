.PHONY: install test serve doctor
install:
	./install.sh
test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests -q
serve:
	./run.sh serve --port 2007
doctor:
	./run.sh doctor
