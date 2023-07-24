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
	cd docs && make html

browse: docs
	$(OPEN) docs/html/index.html

clean:
	$(RM) -r docs/doctrees
	$(RM) -r docs/html
	$(RM) -r docs/source/*/generated
