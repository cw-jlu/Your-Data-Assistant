"""Post-run inspection helpers for ReAct trace.json + benchmark output.

`trace_view` renders a single task's step history for human reading;
`summarize` walks a predictions root and categorizes the run. Both are
pure read paths — they do not modify any artifact and are wired into
`dabench inspect-trace` / `dabench summarize-traces` in cli.py.
"""
