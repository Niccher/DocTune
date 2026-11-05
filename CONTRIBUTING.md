# Contributing to DocTune Studio

Thank you for your interest in contributing to **DocTune Studio**! We welcome contributions ranging from bug fixes and documentation enhancements to new Teacher LLM presets and pipeline optimizations.

Please take a moment to review these guidelines before submitting code.

---

## 1. Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to `domi777nicch@gmail.com`.

---

## 2. Architectural Invariants

To keep the platform clean, predictable, and production-ready, all PRs must respect these foundational design rules:

1. **User `README.md` Rule ($\le 150$ lines)**:
   The root [`README.md`](README.md) is strictly for **operators running the system**. Keep setup instructions brief. In-depth architecture, C4 diagrams, and developer notes belong in [`docs/`](docs/).
2. **Canonical REST API Versioning (`/api/v1/*`)**:
   All new domain REST API endpoints must be prefixed with `/api/v1/*`. We maintain zero backward-compatibility aliases (`/api/*` routes must return 404). Only operational probes (`/health`) and streaming sockets (`/ws/*`) reside at root.
3. **Decoupled Container Dependencies**:
   The web dashboard (`Dockerfile.web`) must remain lightweight and fast. Never add heavy ML dependencies (`torch`, `transformers`, `accelerate`, `bitsandbytes`) to `requirements-web.txt`. Heavy training and data prep run inside the Celery worker (`Dockerfile.worker`).
4. **Multi-Tier Hardware Clamping**:
   Never allow raw user input to allocate unchecked CPU threads or GPU counts. All resource modifications must pass through [`ConfigService`](web/services/config_service.py) to clamp parameters to physical host limits.

---

## 3. Local Development Setup

### Native Python Environment (Without Docker)

```bash
# 1. Clone repository
git clone https://github.com/Niccher/DocTune.git
cd DocTune

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements-web.txt
# Optional: install ML pipeline packages if testing fine-tuning locally
pip install -r requirements.txt

# 4. Copy configuration
cp .env.example .env

# 5. Launch local Web Studio (default port 80, auto-falls back to 8080 if unprivileged)
python run_web.py
```

### Docker Compose Mode

```bash
# Build and start the full 6-service stack
docker compose up --build -d

# Stream web container logs
docker compose logs -f web
```

---

## 4. Running Automated Tests

Before submitting any Pull Request, ensure that the entire test suite passes cleanly:

```bash
# Run unit and integration tests
.venv/bin/python tests/test_web.py
```

All tests should output:
```text
Running tests directly...
All tests passed successfully!
```

---

## 5. Git Branching & Commit Conventions

### Branch Naming

Use descriptive prefixes for your branch names:
* `feat/add-vllm-support` (New features)
* `fix/prevent-memory-leak` (Bug fixes)
* `docs/update-gcp-deployment` (Documentation)
* `refactor/clean-routes` (Code restructuring)
* `test/add-dataset-tests` (Testing additions)

### Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>(<scope>): <short description>

[optional body explaining motivation and context]

[optional footer(s), e.g., BREAKING CHANGE or Closes #123]
```

**Common types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

---

## 6. Submitting a Pull Request

1. Fork the repository on GitHub and create your branch from `main`.
2. Implement your changes following PEP 8 style standards.
3. Add or update tests in [`tests/test_web.py`](tests/test_web.py).
4. Update documentation in [`docs/`](docs/) if modifying API contracts or architecture.
5. Push your branch and open a Pull Request against `main`.
6. Complete the checklist in our [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
