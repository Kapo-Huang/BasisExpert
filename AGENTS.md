# Repository Instructions

- For changes limited to configuration files, YAML assets, config generators,
  run lists, or batch scripts, implement or generate the requested artifacts
  directly.
- Do not add or run automated tests for those simple changes. In particular,
  do not add pytest, unittest, assertion-based, or batch dry-run tests.
- Do not add fixed generated-count assertions or batch-list preflight checks
  for those simple workflows.
- Preserve normal data, model, training, and file-parsing error handling unless
  the user explicitly requests otherwise.
