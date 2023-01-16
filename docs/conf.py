# This file is execfile()d with the current directory set to its containing dir.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import shutil
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "5.0"

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_immaterial",
    "sphinx_immaterial.kbd_keys",
    "sphinx_immaterial.apidoc.format_signatures",
    "sphinx_immaterial.apidoc.python.apigen",
]

rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight

.. role:: rst(code)
   :language: rst
   :class: highlight

.. role:: css(code)
   :language: css
   :class: highlight

.. role:: dot(code)
   :language: dot
   :class: highlight
"""

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "any"

templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "SuShiE"
copyright = "2022, MancusoLab"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# version: The short X.Y version.
# release: The full version, including alpha/beta/rc tags.
# If you don’t need the separation provided between version and release,
# just set them both to the same value.
try:
    from sushie import __version__ as version
except ImportError:
    version = ""

if not version or version.lower() == "unknown":
    version = os.getenv("READTHEDOCS_VERSION", "unknown")  # automatically set by RTD

release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# -- Options for apigen and type annotations
autodoc_class_signature = "separated"

python_apigen_modules = {
    "sushie.utils": "api/haha.rst",
}

python_apigen_default_groups = [
    (r".*:.*", "Public-members"),
    (r"class:.*", "Classes"),
    (r"method:.*\.__(str|repr)__", "String representation"),
    # ("method:.*", "Methods"),
    # ("classmethod:.*", "Class methods"),
    # (r"method:.*\.__(init|new)__", "Constructors"),
    # (r"method:.*\.[A-Z][a-z]*", "Constructors"),
]

python_apigen_default_order = [
    ("class:.*", -6),
    ("method:.*", -2),
    ("classmethod:.*", -3),
    ("property:.*", -1),
    (r"method:.*\.__(init|new)__", -5),
    (r"method:.*\.[A-Z][a-z]*", -5),
    (r"method:.*\.__(str|repr)__", -4),
]

object_description_options = [
    (
        "py.*",
        dict(
            include_in_toc=False,
            include_fields_in_toc=False,
            wrap_signatures_with_css=True,
        ),
    ),
    ("py.class", dict(include_in_toc=True)),
    ("py.function", dict(include_in_toc=True)),
]

python_apigen_order_tiebreaker = "alphabetical"
python_apigen_case_insensitive_filesystem = False
python_apigen_show_base_classes = False

# simplify typing names (e.g., typing.List -> list)
# simplify Union and Optional types
python_transform_type_annotations_pep585 = False
python_transform_type_annotations_pep604 = True
python_transform_type_annotations_concise_literal = True
python_strip_self_type_annotations = True

python_type_aliases = {
    "jnp.ndarray": "jax.numpy.ndarray",
}

python_module_names_to_strip_from_xrefs = [
    "jax._src.numpy.lax_numpy",
]

# fix namedtuple attribute types
napoleon_use_ivar = True
napoleon_google_docstring = True

python_apigen_rst_prolog = """
.. default-role:: py:obj

.. default-literal-role:: python

.. highlight:: python
"""

# Add any paths that contain templates here, relative to this directory.


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_immaterial"

# The name for this set of Sphinx documents.  If None, it defaults to
html_static_path = ["_static"]
# html_css_files = ["extra_css.css"]
html_last_updated_fmt = ""
html_title = "SuShiE"
html_favicon = "_static/images/favicon.ico"
html_logo = "_static/images/dna.png"

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": "https://mancusolab.github.io/sushie/",
    "repo_url": "https://github.com/mancusolab/sushie/",
    "repo_name": "sushie",
    "repo_type": "github",
    "edit_uri": "blob/main/docs",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "toc.follow",
        "toc.sticky",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "teal",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "teal",
            "accent": "lime",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    # BEGIN: version_dropdown
    "version_dropdown": True,
    "version_info": [
        {
            "version": "https://mancusolab.github.io/sushie",
            "title": "Github Pages",
            "aliases": [],
        },
    ],
    # END: version_dropdown
    "toc_title_is_page_title": True,
    # BEGIN: social icons
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/mancusolab/sushie",
        },
        # {
        #    "icon": "fontawesome/brands/python",
        #    "link": "https://pypi.org/project/sphinx-immaterial/",
        # },
    ],
    # END: social icons
}

# Output file base name for HTML help builder.
htmlhelp_basename = "SuShiE-doc"

# -- General options
# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_emit_warnings = True


# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))

intersphinx_mapping = {
    "python": ("https://docs.python.org/" + python_version, None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

print(f"loading configurations for {project} {version} ...", file=sys.stderr)

# -- Post process ------------------------------------------------------------
import collections


def remove_namedtuple_attrib_docstring(app, what, name, obj, skip, options):
    if type(obj) is collections._tuplegetter:
        return True
    return skip


def autodoc_process_signature(
    app, what, name, obj, options, signature, return_annotation
):
    signature = modify_type_hints(signature)
    return_annotation = modify_type_hints(return_annotation)
    return signature, return_annotation


def modify_type_hints(signature):
    if signature:
        signature = signature.replace("jnp", "~jax.numpy")
    return signature


def setup(app):
    app.connect("autodoc-skip-member", remove_namedtuple_attrib_docstring)
    app.connect("autodoc-process-signature", autodoc_process_signature)
