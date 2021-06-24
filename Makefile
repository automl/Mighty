
.PHONY: test
test:
	pytest -v --cov=mighty test --durations=20
