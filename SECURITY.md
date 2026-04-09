# Security Policy

## Scope

SALOMI is a **research codebase** for transformer quantization experiments. It is not a networked service, authentication system, or production deployment tool. The attack surface is limited to:

- local Python execution of experiment and test scripts,
- optional download of public HuggingFace model checkpoints over HTTPS,
- file I/O for result artifacts (`.json`, `.txt`).

## Supported Versions

| Version | Supported |
|---|---|
| `main` branch | Yes |
| Historical commits before April 2026 | No — see `docs/ARCHIVE.md` |

## Reporting a Vulnerability

If you discover a security issue (e.g., unsafe deserialization, path traversal in file I/O, a dependency with a known CVE that affects this codebase), please:

1. **Do not open a public GitHub issue.**
2. Email the maintainer directly, or open a [GitHub Security Advisory](https://github.com/OrionsLock/SALOMI/security/advisories/new) on this repository.
3. Include: a description of the issue, the affected file(s), and a minimal reproduction if possible.

You can expect an acknowledgement within 7 days and a resolution or public disclosure within 30 days.

## Dependencies

Core runtime dependencies (`numpy`, `torch`, `transformers`, `datasets`) are widely audited open-source packages. Keep them up to date via:

```bash
pip install --upgrade -r requirements.txt
```

`pyopencl` is an optional dependency used only for the experimental OpenCL backend. It is not required for any validated result paths.

## Notes

- No credentials, API keys, or secrets are stored in this repository.
- HuggingFace model downloads use public checkpoints over HTTPS; no authentication is required or stored.
- Result artifact files (`.json`, `.txt`) contain only numerical results and are safe to share publicly.
