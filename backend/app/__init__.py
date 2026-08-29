"""J.A.R.V.I.S. backend package."""

#: The single source of truth for the version. `config.Settings.version`
#: reads this rather than carrying its own copy — the two had drifted, so
#: /api/health reported one number while the settings registry held another.
__version__ = "1.5.0"
