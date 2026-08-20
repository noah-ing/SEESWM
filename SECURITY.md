# Security policy

## Reporting a vulnerability

Please use [GitHub's private vulnerability-reporting flow](https://github.com/noah-ing/SEESWM/security/advisories/new).
If that flow is unavailable, open a public issue that asks only for a private
contact channel. Do not include exploit details, credentials, or private data
in a public issue.

This proof of concept is not a production security boundary. Reports affecting
the current `main` branch are in scope; older snapshots and third-party forks
are not maintained.

## Checkpoint boundary

PyTorch checkpoint files are executable-format-adjacent inputs and should not
be treated as trustworthy merely because they use a `.pt` or `.pth` extension.
The schema-v2 evaluator:

- requires PyTorch 2.10 or newer;
- uses the restricted `weights_only` loader;
- rejects files larger than 512 MiB;
- accepts only known configuration, topology, policy, and state structures;
- records the evaluated file's SHA-256 digest.

These controls reduce risk; they do not make arbitrary third-party checkpoints
safe or prevent every denial-of-service condition. Evaluate only checkpoints
you created or whose digest and provenance you independently verified. Never
fall back to `weights_only=False` for an untrusted file.

The minimum version follows PyTorch's
[restricted-loader security advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
PyTorch's [serialization guidance](https://docs.pytorch.org/docs/main/notes/serialization.html#torch-load-with-weights-only)
also documents the restricted loader's remaining denial-of-service and memory
safety limitations.

No API key, cloud credential, or external service is required to run the test
suite or the local grid-world experiments. Keep local values in ignored `.env`
files and never commit them.
