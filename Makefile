# Common tasks for chessvision. Run `make help` to list them.
# Pass extra CLI args with ARGS, e.g.  make capture ARGS='--fen "..."'
.DEFAULT_GOAL := help
.PHONY: help sync detect view corners capture autolabel train export lint fmt docs docs-serve

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync: ## Install/update dependencies and the package
	uv sync

detect: ## Run the live detection / recording app (gm-detect)
	uv run gm-detect $(ARGS)

view: ## Preview the raw camera stream
	uv run gm-view

corners: ## Board-detection debug (corners + warp)
	uv run gm-corners

capture: ## Capture a labelled dataset (needs ARGS='--fen "..."')
	uv run gm-capture $(ARGS)

autolabel: ## Auto-label saved frames (needs ARGS='--fen "..." --src ...')
	uv run gm-autolabel $(ARGS)

train: ## Fine-tune the piece-detection model
	uv run gm-train

export: ## Export weights to ncnn/onnx (e.g. ARGS='--format onnx')
	uv run gm-export $(ARGS)

lint: ## Lint with ruff
	uv run ruff check .

fmt: ## Format with ruff
	uv run ruff format .

docs: ## Build the documentation site
	uv run --group dev zensical build

docs-serve: ## Serve the documentation locally with live reload
	uv run --group dev zensical serve
