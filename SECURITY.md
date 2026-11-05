# Security Policy

DocTune Studio is designed for handling sensitive enterprise documents and model weights. We take the security and privacy of your data, models, and infrastructure seriously.

---

## 1. Supported Versions

We provide security updates and patches for the following versions:

| Version | Supported          | Status |
| :------ | :----------------: | :----- |
| `1.2.x` | :white_check_mark: | Current Active Release (Canonical `/api/v1/*`) |
| `1.1.x` | :white_check_mark: | Critical Security & Regression Fixes Only |
| `< 1.1` | :x:                | Unsupported — Please upgrade immediately |

---

## 2. Reporting a Vulnerability

If you discover a security vulnerability or sensitive information leakage in DocTune Studio, **please do not open a public GitHub issue**.

Instead, report it responsibly through one of the following channels:

* **GitHub Private Security Advisory**: Navigate to the [Security Advisories](https://github.com/Niccher/DocTune/security/advisories/new) tab on GitHub and click **"Report a vulnerability"**.
* **Security Contact**: Email our lead security maintainer at `domi777nicch@gmail.com` with the subject line `[SECURITY] DocTune Vulnerability Report`.

### What to Include in Your Report

To help us triage and resolve the issue quickly, please provide:
1. A clear description of the vulnerability and its potential impact.
2. Step-by-step reproduction instructions or a minimal Proof of Concept (PoC).
3. The specific component affected (e.g., Teacher Gateway, S3 storage handler, WebSocket stream, API router).
4. Environment details (Docker vs native mode, OS, Python version).

---

## 3. Vulnerability Response Timeline

* **Initial Response**: Within **48 hours**, acknowledging receipt of your report and beginning triage.
* **Assessment & Patch**: Within **7 business days**, providing a timeline for a fix.
* **Public Disclosure**: Coordinated release after the patch is deployed and verified. We gladly credit security researchers in our release notes unless anonymity is requested.

---

## 4. Key Security Architecture Invariants

DocTune Studio is engineered with built-in defenses against common LLM platform vulnerabilities:

* **Strict Credential Masking**: Hugging Face tokens (`HF_TOKEN`) and Teacher API keys are masked in UI inputs and logs, never committed to git, and never echoed back in plaintext API responses.
* **Network & SSRF Isolation**: When running inside Docker containers, all inter-service traffic (Postgres, Redis, MinIO) traverses an isolated internal bridge network (`doctune-net`). Host remapping is strictly constrained to `host.docker.internal`.
* **Resource Clamping (Anti-DoS)**: Hardware inputs (CPU threads, max RAM, batch size, epochs) are auto-clamped to physical host safety bounds in [`ConfigService`](web/services/config_service.py) to prevent host exhaustion.
* **Zero Third-Party Data Transmission by Default**: Documents and synthetic pairs are processed locally unless the operator explicitly configures a commercial Teacher LLM endpoint.
