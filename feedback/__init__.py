"""Meta-feedback capture: detects user critiques of LT's prior response and
persists them for review by Claude Code (via the Demerzel project) so the
persona / system prompt / behavior can be tuned over time.

Mirrors memory/extraction.py shape: cheap keyword pre-filter, then a
thinking-OFF Qwen3.6 confirm pass on borderline matches. Fire-and-forget
from main.py's response-finalize path -- never blocks the response loop.
"""
