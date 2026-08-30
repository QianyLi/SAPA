# Security and responsible release

Please report suspected credential leaks or privacy issues privately to the
maintainers rather than opening a public issue.

Before publishing a checkout, verify that:

- `.env` and all API keys are absent;
- benchmark and Amazon source data may legally be redistributed under their
  original terms;
- downloaded model weights and third-party indexes are redistributed only when
  their licenses allow it;
- generated logs do not contain personal, access-token, or provider metadata.

The repository intentionally ignores local datasets, checkpoints, indexes, and
evaluation outputs. See `docs/OPEN_SOURCE_CHECKLIST.md` for the release steps.
