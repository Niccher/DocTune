"""
web/routers package.
Exports router submodules for clean imports in web/main.py.
"""

from . import (
    admin,
    auth,
    dashboard,
    data_prep,
    datasets,
    experiments,
    inference,
    landing,
    models,
    playground,
    training,
)

__all__ = [
    "admin",
    "auth",
    "dashboard",
    "data_prep",
    "datasets",
    "experiments",
    "inference",
    "landing",
    "models",
    "playground",
    "training",
]
