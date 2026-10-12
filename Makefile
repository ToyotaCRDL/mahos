.PHONY: all format format-check lint install-dev test dq-ext docs browse clean

TEST_QPA_PLATFORM ?= offscreen

ifeq ($(OS),Windows_NT)
    OPEN := "start"
else
    UNAME := $(shell uname -s)
    ifeq ($(UNAME),Linux)
        OPEN := xdg-open
    else
        OPEN := open
    endif
endif

all: format lint test

format:
	ruff format .

format-check:
	ruff format --check .

lint:
	ruff check . --statistics

install-dev:
	python -m pip install -e './pkgs/mahos[dev]' -e ./pkgs/mahos-dq

test:
	@python -c "import mahos, mahos_dq" >/dev/null 2>&1 || make install-dev
	QT_QPA_PLATFORM="$(TEST_QPA_PLATFORM)" python -m pytest --timeout=30

dq-ext:
	cd pkgs/mahos-dq-ext/src/mahos_dq_ext && make

docs:
	sphinx-build -b html docs-src docs

browse:
	$(OPEN) docs/index.html

clean:
	$(RM) -r docs
	$(RM) -r docs-src/*/generated
