# Security Policy

## Reporting a Vulnerability

We take security issues seriously and appreciate responsible disclosure.

If you believe you have found a security vulnerability, **please do not report it in a public GitHub issue**.

Instead, use one of the following private channels:

- Open a **GitHub Security Advisory** for this repository (preferred)
- Or contact the maintainers directly if a private email channel is listed

Please include:
- A clear description of the issue
- Steps to reproduce (if applicable)
- Potential impact
- Any relevant proof-of-concept or logs

We will acknowledge receipt and work to assess the issue as quickly as possible.

## Deployment Hardening

If you run the REST API, the OpenRouter-compatible endpoints or the Gradio UI where other machines can reach them, follow the [Security Hardening Guide](./docs/en/SECURITY_HARDENING.md): bind to loopback or a reverse proxy, set `ACESTEP_API_KEY`, confine training paths with `ACESTEP_ALLOWED_PATHS`, and load only trusted model weights.

## Bug Bounties

At this time, this project does **not** operate a formal bug bounty program.  
However, valid and responsibly disclosed security issues may be acknowledged in release notes or documentation at the maintainers’ discretion.

Thank you for helping keep the project and its users safe.
