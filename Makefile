# Repository-level commands dispatch to the selected Agent project.
.DEFAULT_GOAL := help
.PHONY: help install demo report serve console doctor pools pfm-replay evolve-doctor

help: ## Show repository entrypoints.
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the Drug Agent and Self-Evolve runtime.
	bash ./install.sh agent drug core

demo: ## Run the Drug Agent offline demo from its project directory.
	$(MAKE) -C agents/drug-agent demo

report: ## Print the latest locally generated Drug Agent report.
	$(MAKE) -C agents/drug-agent report

serve: ## Serve the Drug Agent dashboard.
	$(MAKE) -C agents/drug-agent serve

console: ## Serve the Drug Agent LLM console.
	$(MAKE) -C agents/drug-agent console

doctor: ## Inspect the Drug Agent runtime and capabilities.
	$(MAKE) -C agents/drug-agent doctor

pools: ## List the Drug Agent's internal roles and tools.
	$(MAKE) -C agents/drug-agent pools

pfm-replay: ## Recount the public PFM example using standard Python only.
	python examples/pfm_replay/replay.py

evolve-doctor: ## Inspect the Self-Evolve configuration without running a cycle.
	$(MAKE) -C agents/drug-agent evolve-doctor
