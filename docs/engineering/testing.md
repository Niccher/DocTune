# Testing Guide & Quality Assurance

DocTune Studio maintains an automated regression test suite in [`tests/test_web.py`](../../tests/test_web.py).

---

## 1. Running Automated Tests

Run the test suite directly with Python:

```bash
# Execute all automated tests
.venv/bin/python tests/test_web.py
```

Expected output:
```text
Running tests directly...
All tests passed successfully!
```

---

## 2. Test Coverage

The test suite validates:
* Route rendering for all 6 pages (Dashboard, Data Prep, Training, Models, Playground, Admin).
* System telemetry and hardware bounds APIs.
* Settings persistence, factory reset, directory creation checks.
* Dynamic application name updates across views.
* Multi-Tier Boundary Clamping (out-of-bounds CPU/RAM/GPU inputs safely clamped).
* Persistent settings integration with `Config.load()`.
* Dataset preview, playground chat generation, and state lifecycle transitions.
