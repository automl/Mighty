
.PHONY: test
test:
	pytest -v --cov=mighty test --durations=20

.PHONY: doc
doc:
	make -C doc html

.PHONY: clean
clean: clean-data
	make -C doc clean
