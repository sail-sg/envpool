# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for the EnvPool docs."""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import shutil
import subprocess


CPP_API_HEADERS = [
    "envpool/core/array.h",
    "envpool/core/spec.h",
    "envpool/core/dict.h",
    "envpool/core/env_spec.h",
    "envpool/core/env.h",
    "envpool/core/envpool.h",
    "envpool/core/async_envpool.h",
    "envpool/core/py_envpool.h",
]
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
DOXYGEN_BUILD_DIR = os.path.join(DOCS_DIR, "_build", "doxygen")
DOXYGEN_XML_DIR = os.path.join(DOXYGEN_BUILD_DIR, "xml")


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    """Return the project version from the package metadata."""
    with open(os.path.join("..", "envpool", "__init__.py"), "r") as f:
        init = f.read().split()
    return init[init.index("__version__") + 2][1:-1]


# -- Project information -----------------------------------------------------

project = "EnvPool"
copyright = "2022, Garena Online Private Limited"
author = "EnvPool Contributors"

# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "sphinx.ext.autodoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst"]

# The root document.
root_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
spelling_exclude_patterns = ["pages/slides.rst"]
breathe_projects = {"envpool_cpp_api": DOXYGEN_XML_DIR}
breathe_domain_by_extension = {"h": "cpp"}
breathe_default_members = ("members", "undoc-members")

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/images/envpool-logo.png"


def generate_doxygen_xml(_app):
    """Generate the Doxygen XML consumed by Breathe."""
    doxygen = shutil.which("doxygen")
    if doxygen is None:
        raise RuntimeError("doxygen is required to build the C++ API docs")
    os.makedirs(DOXYGEN_BUILD_DIR, exist_ok=True)
    doxyfile = os.path.join(DOXYGEN_BUILD_DIR, "Doxyfile")
    inputs = " \\\n".join(
        os.path.join(PROJECT_ROOT, header) for header in CPP_API_HEADERS
    )
    with open(doxyfile, "w", encoding="utf-8") as f:
        f.write(
            f"""PROJECT_NAME = "EnvPool C++ API"
OUTPUT_DIRECTORY = "{DOXYGEN_BUILD_DIR}"
INPUT = {inputs}
FILE_PATTERNS = *.h
RECURSIVE = NO
GENERATE_HTML = NO
GENERATE_LATEX = NO
GENERATE_XML = YES
XML_OUTPUT = xml
EXTRACT_ALL = YES
EXTRACT_PRIVATE = NO
EXTRACT_STATIC = YES
QUIET = YES
WARN_IF_UNDOCUMENTED = NO
STRIP_FROM_PATH = "{PROJECT_ROOT}"
"""
        )
    subprocess.run([doxygen, doxyfile], check=True, cwd=PROJECT_ROOT)


def setup(app):
    """Register the Sphinx configuration hooks."""
    app.connect("builder-inited", generate_doxygen_xml)
    app.add_js_file("js/copybutton.js")
    app.add_css_file("css/style.css")


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/3/': None}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
# todo_include_todos = False
