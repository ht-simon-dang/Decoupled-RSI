PYTHON ?= .venv/bin/python
.DEFAULT_GOAL := help
.PHONY: help install demo doctor serve
help:
	@printf '%s\n' 'Peptide Task Agent' '  bash install.sh --help  Installation options' '  make install          Install this package into an existing runtime' '  make doctor           Inspect local capabilities' '  make demo             Analyze the supplied library example' '  make serve            Start the local Web UI'
install:
	"$(PYTHON)" -m pip install -e . --no-deps
demo:
	"$(PYTHON)" -m peptide_agent.cli run examples/library.yaml
doctor:
	"$(PYTHON)" -m peptide_agent.cli doctor
serve:
	"$(PYTHON)" -m peptide_agent.cli serve --port 2004
