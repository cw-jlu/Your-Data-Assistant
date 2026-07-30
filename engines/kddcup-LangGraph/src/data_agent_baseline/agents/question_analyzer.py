"""Re-export of the ambiguity analyzer prompt constant.

This module exists for backward compatibility with older code that imports
``QUESTION_ANALYZER_SYSTEM_PROMPT`` directly.
"""

from __future__ import annotations

from data_agent_baseline.agents.ambiguity_analyzer import AMBIGUITY_ANALYZER_SYSTEM_PROMPT

QUESTION_ANALYZER_SYSTEM_PROMPT = AMBIGUITY_ANALYZER_SYSTEM_PROMPT
