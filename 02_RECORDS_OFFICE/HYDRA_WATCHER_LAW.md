# Hydra / Watcher Law

> All visual diffs are evidence. Not all visual diffs are alerts.

## Rule

Visual change is a first-class evidence signal, but it is not automatically an operator alert.

## Signal Classes

- content/text delta
- structured DOM delta
- semantic/claim delta
- visual/layout delta
- archive/prior-version delta

## Watcher Outcomes

- `ignore`
- `record_only`
- `alert`
- `escalate`

## Policy Consequence

The watcher profile decides what matters. A page redesign with unchanged text may be low signal for a public-record source and high signal for a checkout, login, form, disclosure, or evidence-display workflow.

## Canonical Boundary

Evidence capture maximizes memory. Alerting minimizes noise.
