.PHONY: all test lint format docs browse clean

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

test:
	pytest --timeout=10

lint:
	flake8 . --show-source --statistics

format:
	black .

docs:
	sphinx-build -b html docs_src docs

browse:
	$(OPEN) docs/index.html

clean:
	$(RM) -r docs
	$(RM) -r docs_src/*/generated
