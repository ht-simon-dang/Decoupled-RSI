# Runtime entrypoints. Invoke from the repository root.
CONDA ?= conda
CONDA_ENV ?= drug-agent
RUN ?= $(CONDA) run --no-capture-output -n $(CONDA_ENV)
PORT ?= 2003
LLM_MODEL ?= qwen3.8:27b
EXTERNAL_DIR ?= .external
.DEFAULT_GOAL := help
export DRUGAGENT_ALLOW_NETWORK ?= 0

.PHONY: help install env bootstrap demo report serve console doctor pools schemas fixtures
.PHONY: pfm-install pfm-clone external-clone external-status side-env pocket
.PHONY: aizynth-install boltz-install unimol-install vina-install synflownet-install aizynth-models
.PHONY: evolve-doctor evolve-analyze

help: ## Show runtime and optional-install commands.
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create/update the main runtime environment and install drugagent.
	CONDA_EXE="$(CONDA)" DRUGAGENT_CONDA_ENV="$(CONDA_ENV)" bash ./install.sh core

env: install ## Alias for the main runtime installation.

bootstrap: install ## Install the runtime and regenerate the synthetic offline fixtures.
	$(RUN) python fixtures/generate_fixtures.py

demo: ## Run the synthetic offline drug-discovery example.
	$(RUN) python scripts/run_demo.py

report: ## Print the latest locally generated run report.
	$(RUN) python scripts/show_latest_report.py

serve: ## Serve the local dashboard and API.
	$(RUN) drugagent serve --host 127.0.0.1 --port "$(PORT)"

console: ## Serve the LLM task console; requires a separately configured Ollama model.
	$(RUN) drugagent console --host 127.0.0.1 --port "$(PORT)" --model "$(LLM_MODEL)"

doctor: ## Inspect the installed runtime and optional capabilities.
	$(RUN) drugagent doctor

pools: ## List Task Agent roles and available tools.
	$(RUN) drugagent agents
	$(RUN) drugagent tools

schemas: ## Regenerate public interface schemas.
	$(RUN) python -m drug_agent.cli.export_schemas schemas

fixtures: ## Regenerate synthetic offline example inputs.
	$(RUN) python fixtures/generate_fixtures.py

pfm-install: ## Install PFM dependencies only; compilation may be required.
	CONDA_EXE="$(CONDA)" DRUGAGENT_CONDA_ENV="$(CONDA_ENV)" bash ./install.sh pfm

pfm-clone: ## Fetch the pinned PFM source checkout separately.
	DRUGAGENT_EXTERNAL_DIR="$(EXTERNAL_DIR)" bash ./install.sh pfm-source

external-clone: ## Fetch ONE upstream checkout: make external-clone TOOL=REINVENT4.
	@test -n "$(TOOL)" || { echo "Set TOOL to a name from ./install.sh list."; exit 2; }
	DRUGAGENT_EXTERNAL_DIR="$(EXTERNAL_DIR)" bash ./install.sh external "$(TOOL)"

external-status: ## Inspect which optional tools can run.
	$(RUN) drugagent tools

side-env: ## Install ONE side environment: make side-env TOOL=vina.
	@test -n "$(TOOL)" || { echo "Set TOOL to aizynth, boltz, synflownet, unimol, or vina."; exit 2; }
	CONDA_EXE="$(CONDA)" bash ./install.sh side-env "$(TOOL)"

aizynth-install: ## Install the AiZynthFinder environment; no model download.
	CONDA_EXE="$(CONDA)" bash ./install.sh side-env aizynth

boltz-install: ## Install the Boltz environment; checkpoints are separate.
	CONDA_EXE="$(CONDA)" bash ./install.sh side-env boltz

unimol-install: ## Install the Uni-Mol environment; checkpoints are separate.
	CONDA_EXE="$(CONDA)" bash ./install.sh side-env unimol

vina-install: ## Install the Vina/OpenMM environment.
	CONDA_EXE="$(CONDA)" bash ./install.sh side-env vina

synflownet-install: ## Install the CUDA-specific SynFlowNet dependency environment.
	CONDA_EXE="$(CONDA)" bash ./install.sh side-env synflownet

aizynth-models: ## Explicitly download AiZynthFinder public model data.
	mkdir -p "$(EXTERNAL_DIR)/aizynthfinder-data"
	cd "$(EXTERNAL_DIR)/aizynthfinder-data" && $(CONDA) run --no-capture-output -n drug-agent-aizynth download_public_data .

pocket: ## Prepare a PFM pocket: make pocket PDB=input.pdb LIGAND=ABC OUT=pocket.pdb.
	@test -n "$(PDB)" -a -n "$(LIGAND)" -a -n "$(OUT)" || { echo "Set PDB, LIGAND, and OUT."; exit 2; }
	$(RUN) python scripts/prepare_pocket.py "$(PDB)" --ligand "$(LIGAND)" -o "$(OUT)"

evolve-doctor: ## Inspect Self-Evolve Agent configuration without starting a cycle.
	$(RUN) drugagent self-evolve doctor --config configs/self_evolve.yaml

evolve-analyze: ## Analyze newly generated local task experience.
	$(RUN) drugagent self-evolve analyze --config configs/self_evolve.yaml

# The public snapshot omits the private regression suite. baseline/step/run
# retain their original gate 0 and cannot pass until that suite is supplied.
