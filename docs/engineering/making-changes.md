# Making Changes & Contribution Matrix

Guidance for engineers modifying features or adding endpoints to DocTune Studio.

---

## 1. Task Matrix

| If you want to… | Touch these files |
| :--- | :--- |
| **Add a new fine-tuning algorithm** | `pipeline/` (`train_<method>.py`), `web/routers/training.py`, `web/templates/training.html` |
| **Add a new system configuration knob** | `config/settings.json`, `web/services/config_service.py`, `web/templates/admin.html` |
| **Add a new database model / column** | `web/db/models.py`, `web/db/session.py`, `docs/architecture/data-and-storage.md` |
| **Modify branding or UI layout** | `web/templates/base.html`, `web/templates_config.py` |
| **Adjust hardware boundary clamping** | `web/services/system_service.py`, `web/services/config_service.py`, `tests/test_web.py` |
| **Add an API endpoint** | `web/routers/<domain>.py`, `docs/api/contract.md`, `docs/api/openapi.yaml` |

---

## 2. Definition of Done (DoD)

Before opening a pull request or merging changes into `master`:

- [ ] Code is implemented cleanly with docstrings and type annotations.
- [ ] Any new configuration option has a fallback default in `FACTORY_DEFAULTS`.
- [ ] Automated tests in `tests/test_web.py` pass (`python tests/test_web.py`).
- [ ] If a user-facing behavior changed, update `README.md` ($\le 150$ lines).
- [ ] If an API contract changed, update `docs/api/contract.md` and `docs/api/openapi.yaml`.
- [ ] Add a bullet to `CHANGELOG.md` following Semantic Versioning.
