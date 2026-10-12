# Repository Guidelines

## Project Structure

This repository is a Python monorepo. The core package is in `pkgs/mahos/`, with additional
packages in `pkgs/mahos-dq/` and `pkgs/mahos-dq-ext/`. Each package uses a src layout.
Tests are under `tests/`, and examples and sample configurations are in `examples/`.
Sphinx sources are in `docs-src/`, and generated HTML is written to `docs/`.
Ignore these directories when inspecting or editing:

- `docs/`: generated documentation.
- `misc/`: files unrelated to the code or documentation.

## Build, Test, and Development Commands

- `make install-dev` (or `pip install -e './pkgs/mahos[dev]' -e ./pkgs/mahos-dq`):
  install `mahos` and `mahos-dq` for development and testing.
- `pip install -e ./pkgs/mahos-dq-ext`: install the optional C extension package
  when its features are needed. The `-e` option may be omitted.
- `make test` (or `QT_QPA_PLATFORM=offscreen pytest --timeout=30`):
  run the test suite with timeouts.
- `make lint`: run Ruff lint checks.
- `make format`: format Python code with Ruff.
- `make format-check`: check Python formatting with Ruff without modifying files.
- `make docs`: build Sphinx HTML docs from `docs-src/` into `docs/`.
- `make browse`: open the built docs in a browser.

## Coding Style and Naming Conventions

Python code follows the Ruff formatter and linter with a line length of 99.
Indentation is 4 spaces. Use PEP 8 naming: `snake_case` for functions and
variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
Generated UI code in `pkgs/mahos/src/mahos/gui/ui` and `pkgs/mahos-dq/src/mahos_dq/gui/ui`
is excluded from formatting and linting.

## Input Validation

- Prefer the shared validation APIs over adding local helpers for common type, finiteness,
  sign, sequence, and ordering checks.
- For standalone values, use `import mahos.util.validation as V` and the module-level
  `V.check_*()` functions.
- For static configuration in `self.conf`, inherit from `ConfAccessorMixin` and use its
  `_conf_*()` methods.
- For dynamic parameter mappings, use `ParamAccessor` and its validated access methods.
- Keep domain-specific constraints and cross-field validation in the domain code.

## Docstrings

- Write all docstrings in reStructuredText (reST).
- Keep one-line dosctrings on one line, including the opening and closing triple quotes
  (like ``"""Summary."""``).
- For multi-line docstrings, put the summary on the first line, followed by a blank line and
  the detailed description. Put the closing triple quotes on their own line after a blank line.
- Insert exactly one blank line immediately after a docstring before the following code.
- Provide comprehensive top-level class docstrings for all `Instrument`, `Node`,
  `GUINode`, and `Message` classes.
- For measurement Nodes with exactly one Worker, the Worker docstring may be omitted if the
  Worker behavior is documented in the Node docstring.
- For measurement Nodes with multiple Workers, each Worker should have its own docstring, and
  the Node docstring should link to those Worker docstrings.
  Use `pkgs/mahos-dq/src/mahos_dq/meas/odmr.py` (ODMR Node) as a reference pattern.
- For multi-line top-level class docstrings, structure the detailed description in this order:
  - When applicable, a few paragraphs describing behavior and important requirements.
  - For the `Instrument`, `Node`, and `GUINode` classes, a list of static configuration keys in
    `self.conf` (loaded from `conf.toml`), using `:param:` and `:type:` directives.
    - Express nested configuration keys with dot notation (like `level1_key.level2_key`).
  - For `Message` classes, list attributes with `:ivar:` directives.
  - Optional minor notes may follow.
- For `Message` classes derived from `enum.Enum` (including `State` subclasses),
  use a one-line docstring only and omit `:ivar:` listings.

## Testing

Tests use pytest. Name test files `test_*.py`, and focus new unit tests on the module you change.
Run targeted tests with `pytest tests/mahos/test_x.py`. Use `tests/conf.toml` as a sample test
configuration.

## Commit Messages

Use a short, imperative first line for commit subjects. Scope prefixes are
recommended when applicable:

- `docs:` for documentation-only patches.
- `ClassName:` when the patch primarily affects a specific class.
- `module_name:` when the patch is module-wide.

For large patches, add a commit body with a bullet list describing the key changes.
Keep the subject concise and lowercase where natural.
Do not use backticks in commit subjects or bodies.

## GUI Implementation and Tests

- Keep GUI implementations simple. When introducing shared helper logic, prefer module-level
  functions or ordinary Python objects unless QObject-specific behavior is required.
- Avoid QObject-based helpers that introduce unnecessary object references or signal/slot
  connections; these can make GUI teardown nondeterministic.
- Avoid excessive GUI component tests built around monkeypatching or ad hoc mocks.
  Prefer end-to-end tests (`tests/mahos/test_gui_e2e.py` or `tests/mahos-dq/test_gui_e2e_dq.py`)
  or unit tests focused on important logic.

## CLI Completion Performance

- Keep CLI parser-building paths lightweight for `argcomplete`.
- Do not import heavy optional or runtime libraries at module import time in CLI modules loaded
  while building completion parsers (for example, plotting, shell, or data commands).
- Move heavy imports into execution-time functions (`main()` or command handlers) unless they
  are required to define parser arguments.
