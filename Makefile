
# NOTE: Used on linux, limited support outside of Linux
#
# A simple makefile to help with small tasks related to development of Mighty
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install-dev install check format pre-commit clean build clean-doc clean-build test doc publish

help:
	@echo "Makefile Mighty"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with ruff"
	@echo "* typing           to type check the code with mypy"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* clean            to clean the dist and doc build files"
	@echo "* build            to build a dist"
	@echo "* test             to run the tests"
	@echo "* doc              to generate and view the html files"
	@echo "* publish          to help publish the current branch to pypi"

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags
PIP ?= python -m pip
MAKE ?= make
PRECOMMIT ?= pre-commit
RUFF ?= ruff
MYPY ?= mypy

DIR := ${CURDIR}
DIST := ${CURDIR}/dist
DOCDIR := ${CURDIR}/docs
INDEX_HTML := file://${DOCDIR}/html/build/index.html

install-dev:
	$(PIP) install pufferlib==1.0.0
    $(PIP) install dacbench==0.3.0 torchvision ioh
    $(PIP) install -e ".[dev,carl]"

install:
	$(PIP) install -e ".[examples]"


# pydocstyle does not have easy ignore rules, instead, we include as they are covered
check: 
	ruff format --check mighty test
	ruff check mighty test

pre-commit:
	$(PRECOMMIT) run --all-files

format: 
	$(RUFF) format --silent mighty test
	$(RUFF) check --fix --silent mighty test --exit-zero
	$(RUFF) check --fix mighty test --exit-zero

typing:
	$(MYPY) mighty

test:
	$(PYTEST) -v --cov=mighty test --durations=20 --cov-report html

clean-doc:
	$(MAKE) -C ${DOCDIR} clean

clean-build:
	rm -rf ${DIST}

# Clean up any builds in ./dist as well as doc
clean: clean-doc clean-build

# Build a distribution in ./dist
build:
	$(PYTHON) -m build

doc:
	$(MAKE) -C ${DOCDIR} docs
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean-build build
	$(PIP) install twine
	$(PYTHON) -m twine upload --verbose --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following line:"
	@echo "pip install --index-url https://test.pypi.org/simple/ mighty"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo "python -m twine upload dist/*"