"""Sphinx configuration for the ConflictxGame documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "ConflictxGame"
author = "ConflictxGame contributors"
copyright = "2026, ConflictxGame contributors; portions 2025 Andy Liu (ConflictScope)"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Heavy / platform-specific dependencies are mocked so the docs build anywhere
# (vllm and fcntl are unavailable on Windows; the rest are optional installs).
# matplotlib/seaborn are NOT mocked: analyze_experiment configures matplotlib
# at import time in ways a mock cannot satisfy, so they must be installed.
autodoc_mock_imports = [
    "vllm",
    "openai",
    "anthropic",
    "sentence_transformers",
    "wandb",
    "statsmodels",
    "sklearn",
    "choix",
    "fcntl",
]

autodoc_member_order = "bysource"
# "undoc-members" is deliberately off: dataclass attributes are documented via
# the Google-style ``Attributes:`` sections (napoleon), and including the bare
# members as well produces duplicate-object warnings.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = []
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navbar_align": "left",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
