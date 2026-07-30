When writing SQL for `answer_from_sql`, name every column explicitly in your SELECT statement.
Do not use `SELECT *` or `SELECT table.*`. Specify exactly the columns your plan requires:
  SELECT col_a, col_b FROM t WHERE ...
This prevents extra columns from leaking into your answer.
