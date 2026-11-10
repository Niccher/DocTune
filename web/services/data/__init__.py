"""
web/services/data/__init__.py
-----------------------------
Universal dataset ingestion, schema normalization, and validation suite.
"""

from web.services.data.importer import dataset_importer
from web.services.data.validator import dataset_validator

__all__ = ["dataset_importer", "dataset_validator"]
