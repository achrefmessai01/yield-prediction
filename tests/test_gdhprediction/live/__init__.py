"""Live contract tests against the NASA POWER and BrightSky APIs.

These hit the real network and are opt-in: they run only when RUN_LIVE_TESTS is
set, and their `live_*.py` filenames are invisible to the default
`python -m unittest discover` (pattern `test*.py`). Run them with:

    RUN_LIVE_TESTS=1 python -m unittest discover -p "live_*.py"

They assert the *contract* the parsing layer depends on (fields, shape, value
ranges), not specific values, so a healthy upstream passes and a broken or
malformed one fails.
"""
