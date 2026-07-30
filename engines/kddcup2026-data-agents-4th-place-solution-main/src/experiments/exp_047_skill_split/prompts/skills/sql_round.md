In SQL, apply ROUND() only at the outermost SELECT level. Never round inside subqueries
or CTEs — intermediate rounding propagates precision loss. Example:
  SELECT ROUND(AVG(price), 2) FROM items   -- correct: rounds final result
  SELECT AVG(ROUND(price, 2)) FROM items   -- wrong: rounds before aggregating
