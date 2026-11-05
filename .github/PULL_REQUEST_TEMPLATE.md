## Description of Changes

<!-- Provide a brief, clear summary of what this Pull Request changes and why. -->

---

## Type of Change

- [ ] :sparkles: **New Feature** (non-breaking change adding functionality)
- [ ] :bug: **Bug Fix** (non-breaking change fixing an issue)
- [ ] :recycle: **Refactor** (restructuring without behavior changes)
- [ ] :zap: **Performance Improvement** (optimization in throughput or memory)
- [ ] :books: **Documentation** (updates to README or docs/)
- [ ] :boom: **Breaking Change** (fix or feature that alters existing API contracts)

---

## Architectural Invariants Checklist

Please verify that your changes adhere to DocTune Studio's core design rules:

- [ ] **Canonical REST APIs**: All new endpoints are prefixed with `/api/v1/*`. No unversioned `/api/*` routes are introduced.
- [ ] **README Conciseness**: Root [`README.md`](README.md) remains $\le 150$ lines (in-depth documentation added to `docs/`).
- [ ] **Resource Clamping**: Any user-adjustable compute options pass through hardware safety checks.
- [ ] **Secret Isolation**: No tokens (`HF_TOKEN`), API keys, or `.env` files are tracked or committed.
- [ ] **Dependency Decoupling**: Heavy ML libraries (`torch`, `peft`) are NOT added to `requirements-web.txt`.

---

## Testing & Verification

- [ ] Automated test suite executed: `.venv/bin/python tests/test_web.py`
- [ ] Output confirmed: `All tests passed successfully!`
- [ ] Manual verification in browser / container:
  <!-- Describe manual test steps performed -->

---

## Related Issues

<!-- If applicable, link related issues (e.g., Closes #12, Fixes #34) -->
