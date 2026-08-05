# Contributing

Contributions to SuShiE are welcome, including bug reports, documentation
improvements, tests, and code changes. All contributors must follow the
[code of conduct](conduct.md).

## Report an issue

Search the [issue tracker](https://github.com/mancusolab/sushie/issues),
including closed issues, before opening a new report. Include your operating
system, Python version, relevant input format, and the smallest reproducible
example you can provide.

## Improve the documentation

The documentation is written in Markdown and built with
[Zensical](https://zensical.org/). Documentation changes follow the same pull
request workflow as code changes.

Install the development environment and run a strict build:

```bash
uv sync --extra dev
uv run zensical build --clean --strict
```

Preview the generated site at <http://localhost:8000>:

```bash
uv run python -m http.server --directory site
```

The API reference imports `sushie` and its dependencies. If API collection
fails, first make sure `uv sync --extra dev` completed successfully.

## Contribute code

### Create an environment

Clone your fork, then install the development and testing dependencies:

```bash
git clone git@github.com:YourLogin/sushie.git
cd sushie
uv sync --extra dev --extra testing
uv run pre-commit install
```

You can instead use an editable `pip` environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,testing]"
```

### Implement and verify your changes

Create a focused feature branch:

```bash
git checkout -b my-feature
```

Add tests and documentation for user-visible changes, and add docstrings for
new public functions, modules, and classes. Add yourself to
[the contributors list](authors.md) when appropriate.

Run the relevant tests and static checks before submitting the change:

```bash
uv run --extra testing pytest -p no:capture
uv run ruff check sushie tests data
uv run ruff format --check sushie tests data
uv run ty check sushie tests data
uv run zensical build --clean --strict
```

### Submit your contribution

Commit the focused changes, push your branch, and open a
[pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request):

```bash
git add <modified-files>
git commit
git push -u origin my-feature
```

## Troubleshooting development builds

- Fetch upstream tags if `git describe --abbrev=0 --tags` reports an
  unexpected version. Forks used in CI must also contain the relevant tags.
- Recreate the environment with `uv sync --reinstall` if installed metadata is
  stale.
- Pass `--pdb` to pytest to enter the debugger after a failure.

## Maintainer releases

Before a release, maintainers should verify the full test suite, tag the
release from `main`, build the distributions with `uv build`, inspect their
version and contents, and publish through the project's configured PyPI
workflow.
