import automl_sphinx_theme

from mighty import author, version, name, copyright

options = {
    "copyright": copyright,
    "author": author,
    "version": version,
    "name": name,
    "html_theme_options": {
        "github_url": "https://github.com/automl/Mighty",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
    "autosummary_generate": False,
    "exclude_patterns": ["static", "templates", "archive"]
}

automl_sphinx_theme.set_options(globals(), options)

