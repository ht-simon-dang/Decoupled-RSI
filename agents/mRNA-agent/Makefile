MRNA_DATA_DIR ?= $(HOME)/.cache/decoupled-rsi/mrna
export MRNA_DATA_DIR
PYTHON ?= $(MRNA_DATA_DIR)/venv/bin/python
.DEFAULT_GOAL := help
.PHONY: help install bootstrap optional-models doctor demo serve benchmark
help:
	@printf '%s\n' 'mRNA Task Agent' '  bash install.sh --help  Installation options' '  make bootstrap        Install runtime and download model assets' '  make install          Install this package into an existing runtime' '  make optional-models  Download optional backend assets' '  make doctor           Inspect local capabilities' '  make demo             Run the HBB example' '  make serve            Start the local Web UI' '  make benchmark        Run local strategy review'
install:
	"$(PYTHON)" -m pip install -e . --no-deps
bootstrap:
	bash install.sh
doctor:
	"$(PYTHON)" -m mrna_agent.cli doctor
demo:
	"$(PYTHON)" -m mrna_agent.cli run examples/hbb.json
serve:
	"$(PYTHON)" -m mrna_agent.cli serve --port 2005
benchmark:
	"$(PYTHON)" -m mrna_agent.cli benchmark --evolve
optional-models:
	"$(PYTHON)" scripts/bootstrap_optional.py
