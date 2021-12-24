import datetime
import doctest
import sphinx_rtd_theme
import condorml

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

autosummary_generate = True
templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"

author = "Praveen Chandar"
project = "condorml"
copyright = f"{datetime.datetime.now().year}, {author}"

version = condorml.__version__
release = condorml.__version__

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
    #'tensorflow': ('https://www.tensorflow.org/api_docs', None),
}

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "navigation_depth": 2,
}

html_logo = "_static/img/logo.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
rst_context = {"condorml": condorml}

add_module_names = False


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            "__init__",
            "__repr__",
            "__weakref__",
            "__dict__",
            "__module__",
        ]
        return True if name in members else skip

    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect("autodoc-skip-member", skip)
    app.connect("source-read", rst_jinja_render)