When the question uses superlative words (lowest, highest, least, most, minimum, maximum,
cheapest, largest, smallest, etc.), do NOT use LIMIT 1. Find the extreme value first,
then filter back to return ALL rows that share that value:
  SELECT * FROM t WHERE col = (SELECT MIN(col) FROM t)
This handles ties correctly. Only use LIMIT 1 when the question explicitly asks for
"the single" or "any one" result.
