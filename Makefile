.PHONY: install lint test eval playground infra-single infra-cicd deploy

install:
	uv sync --frozen --extra lint

lint:
	uv run ruff check app tests/unit tests/eval
	uv run codespell

test:
	uv run pytest tests/unit -v

eval:
	agents-cli eval run --config tests/eval/eval_config.yaml

playground:
	agents-cli playground

infra-single:
	agents-cli infra single-project

infra-cicd:
	agents-cli infra cicd

deploy:
	agents-cli deploy
