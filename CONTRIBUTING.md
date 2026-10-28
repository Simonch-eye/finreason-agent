# Contributing to FinReason-Agent

Thank you for your interest in contributing to **FinReason-Agent**! We welcome contributions across all areas: financial parsing, deterministic sandbox algorithms, cognitive routing, forensic skill libraries, and benchmark evaluations.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/finreason-agent.git
   cd finreason-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies in editable mode:**
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   pip install -r requirements.txt  # or install optional tools
   ```

---

## 🧪 Testing & Verification

Before submitting a Pull Request, ensure that all tests and linting checks pass:

1. **Run full regression test suite (55 test cases):**
   ```bash
   python -m pytest
   ```

2. **Run code quality & linting checks:**
   ```bash
   ruff check finreason tests
   ```

3. **Verify benchmark scorecard:**
   ```bash
   finreason eval --suite 20
   ```

---

## 📐 Pull Request Guidelines

1. **Deterministic Verification**: Any new financial calculation or ratio must be backed by deterministic DuckDB/Python code in the sandbox rather than LLM mental arithmetic.
2. **Backward Compatibility**: Ensure all aliases (`FinancialPDFParser`, `ExtractedTable`, `ParseResult`) and public APIs remain compatible.
3. **Commit Messages**: Write clear, descriptive commit messages following the Conventional Commits specification (e.g., `feat: add cash flow articulation skill`, `fix: handle multiline table header continuation`).

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
