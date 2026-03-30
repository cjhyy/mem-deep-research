# Contributing to Mem Deep Research

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/cjhyy/mem-deep-research.git
cd mem-deep-research
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

## Pull Request Process

1. Fork the repository and create your branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps, expected behavior, and actual behavior
- For security vulnerabilities, please email chenjunhong54321@163.com directly

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
