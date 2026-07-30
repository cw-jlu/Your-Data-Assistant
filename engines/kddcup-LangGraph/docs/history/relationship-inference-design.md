# Relationship Inference Design

## Goal

`inspect_all_schema` should help the agent see likely join paths across CSV, JSON, and SQLite assets without flooding the context with raw data. The relationship detector therefore treats field names as hints, not proof. A relationship is emitted only after candidate fields pass type checks and value-overlap validation.

The design deliberately prefers precision over recall: missing a weak relationship is better than giving the agent a plausible but false foreign key.

## Pipeline

1. Build a unified structural catalog for CSV, JSON, and SQLite files using the existing schema scanners.
2. Collect field references with asset path, table name, field name, inferred type, row count, cardinality, and SQLite primary-key metadata.
3. Generate relationship candidates from explicit SQLite foreign keys and conservative naming rules.
4. Profile only candidate fields, not every pair of fields.
5. Validate candidates by comparing source values against target key values.
6. Emit relationships whose confidence passes the threshold.

## Candidate Generation

The detector first identifies source fields that look like references:

- Names containing `id`, `key`, or `code`.
- Names such as `user_id`, `OwnerUserId`, `PostId`, `records.OwnerUserId`.
- Generic source fields named only `Id` are rejected because `Id -> Id` across unrelated entities is usually false.
- Metric or content fields such as `count`, `score`, `age`, `amount`, `date`, `time`, `text`, `body`, and `title` are rejected.

Target fields must look like entity keys:

- SQLite primary keys.
- Plain key fields such as `Id`, `id`, `key`, or `code`.
- High-uniqueness identifier fields where cardinality is close to row count.

Name matching is conservative. For example, `OwnerUserId` can match `users.Id` because the remaining semantic token is `user`, while `PostTypeId` does not match `posts.Id` because the extra `type` token indicates a different lookup table.

## Data Validation

For each candidate pair, the detector reads values from the two fields and computes:

- `source_non_null_count`
- `source_distinct_count`
- `target_distinct_count`
- `matched_source_row_ratio`
- `matched_source_distinct_ratio`
- `target_uniqueness_ratio`

Inferred relationships require enough evidence:

- At least 3 non-null source rows.
- At least 2 distinct source values.
- At least 95% matched source rows or 95% matched source distinct values.
- Compatible value types.

SQLite explicit foreign keys are always eligible, but their evidence still includes the same value-overlap metrics.

## Confidence

Confidence combines:

- Name evidence.
- Value-overlap ratio.
- Target uniqueness.
- Whether the relationship is an explicit SQLite foreign key.
- Whether value profiling hit the distinct-value cap.

Only relationships with confidence at least `0.80` are emitted, except explicit SQLite foreign keys, which receive at least `0.95` confidence.

## False Positive Controls

The main safeguards are:

- Reject generic `Id -> Id` matches across unrelated assets.
- Reject metric/content fields before value comparison.
- Require `PostTypeId`-style modifier tokens to match the target entity, preventing `PostTypeId -> posts.Id`.
- Require high source-to-target value overlap.
- Lower confidence if a field exceeds the profiling cap.
- Avoid lower-casing string values during comparison, preserving case-sensitive codes.

## Output Shape

Each relationship uses this structure:

```json
{
  "source": {
    "asset_path": "orders.csv",
    "table": null,
    "fields": ["user_id"]
  },
  "target": {
    "asset_path": "users.csv",
    "table": null,
    "fields": ["id"]
  },
  "relationship_type": "foreign_key",
  "cardinality": "many_to_one",
  "confidence": 0.97,
  "evidence": {
    "explicit_sqlite_foreign_key": false,
    "name_match": "source key tokens match target entity/key tokens",
    "source_non_null_count": 100,
    "source_distinct_count": 10,
    "target_distinct_count": 10,
    "matched_source_row_ratio": 1.0,
    "matched_source_distinct_ratio": 1.0,
    "target_uniqueness_ratio": 1.0,
    "value_profile_capped": false
  }
}
```

This gives the agent both the proposed join path and enough evidence to judge whether to trust it.
