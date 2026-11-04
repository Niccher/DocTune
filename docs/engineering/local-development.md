# Local Engineering Development Setup

This document guides engineers who are editing code directly on their workstation without running the complete Docker stack.

---

## 1. Prerequisites

* **OS**: Linux (Ubuntu 22.04+ / Debian 12 / Arch), macOS, or WSL2.
* **Python**: 3.11 or 3.12 with `python3-venv` and `pip`.
* **C/C++ Compiler**: `gcc` / `g++` (required for certain C-extensions like `bitsandbytes` or `sentencepiece`).

---

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Niccher/Fine-Tuning.git
cd "Fine-Tuning"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install web dependencies and ML dependencies
pip install -r requirements-web.txt
pip install -r requirements.txt

# Run the test suite
python tests/test_web.py

# Launch development web server
python run_web.py
```
