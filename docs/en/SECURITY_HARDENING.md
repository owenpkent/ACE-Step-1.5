# Security Hardening Guide

This guide is for anyone running ACE-Step 1.5 as a service: the REST API (`acestep.api_server`), the OpenRouter-compatible endpoints, or the Gradio UI reachable by other machines. If you only open the Gradio UI on your own machine at `http://localhost:7860`, the defaults are already appropriate and you can skip to [Model weights](#model-weights-and-remote-code).

ACE-Step is a single-tenant tool. It has one shared API key, no per-user accounts, no rate limiting, and the training routes read and write the filesystem of the host it runs on. Treat the API server as a privileged process and expose it deliberately.

---

## 1. Default network posture

| Component | Default bind | Notes |
| :--- | :--- | :--- |
| Gradio UI (`acestep`) | `127.0.0.1:7860` | `--server-name 0.0.0.0` or `SERVER_NAME=0.0.0.0` opens it to the network. The UI has no authentication of its own. |
| REST API (`acestep.api_server`, `run_api_server.sh`) | `127.0.0.1:8001` | `ACESTEP_API_HOST=0.0.0.0` or `--host 0.0.0.0` opens it to the network. |
| OpenRouter-compatible API | same as the server that hosts it | Mounted inside the REST API server under `/v1/...`, and also available standalone via `run_openrouter_api_server.sh`. |
| TensorBoard (started by `/v1/training/...`) | same host as `ACESTEP_API_HOST` | Previously started with `--bind_all`; it now binds the API host only. |
| Docker image | `0.0.0.0` inside the container | Publish the port to loopback only unless you need LAN access: `-p 127.0.0.1:8001:8001`. |

Keep every listener on loopback unless another machine must reach it. If it must, put the API behind a reverse proxy that terminates TLS and restricts source addresses, and set an API key (next section).

---

## 2. Require an API key

Without `ACESTEP_API_KEY`, every API route accepts anonymous requests, including the training and dataset routes that write files. Set a long random key before exposing the server:

```bash
export ACESTEP_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m acestep.api_server
```

Clients send it as `Authorization: Bearer <key>`. The key is compared in constant time. The same key protects the OpenRouter-compatible `/v1/chat/completions`, `/v1/models` and related routes mounted on the API server. In earlier builds those routes were mounted without the key check even when `ACESTEP_API_KEY` was set, so an operator who relied on the key alone was still exposing them. Upgrade before relying on the key.

The standalone OpenRouter server (`run_openrouter_api_server.sh`) uses `OPENROUTER_API_KEY` instead. See [OpenRouter API](./Openrouter_API_DOC.md#authentication).

The Gradio UI has no API key. If the UI must be reachable by others, protect it with a reverse proxy that handles authentication, or use `--share` only for short sessions and stop the server afterwards.

---

## 3. Confine filesystem paths for training routes

The training, dataset and LoRA routes (`/v1/training/*`, `/v1/dataset/*`, `/v1/lora/load`) take directories and file paths from the request body. Every such path is resolved (symlinks followed) and must sit under one of these roots, or the request fails with `400`:

- the project directory (which already covers `checkpoints/`, `gradio_outputs/` and the default temp directory)
- the working directory the server was started from
- `ACESTEP_TMPDIR`, if set
- each entry of `ACESTEP_ALLOWED_PATHS`

`ACESTEP_ALLOWED_PATHS` is separated by `;` on Windows and `:` elsewhere:

```bash
# Linux / macOS
export ACESTEP_ALLOWED_PATHS=/data/datasets:/data/loras
# Windows (PowerShell)
$env:ACESTEP_ALLOWED_PATHS = "D:\datasets;D:\loras"
```

Two practical consequences:

- Start the server from the project directory, not from your home directory. The startup working directory is an allowed root, so starting from `~` would allow the whole home directory.
- Destructive operations (the export route removes and recreates `export_path`) refuse a path that equals an allowed root, so you cannot wipe a root by passing it directly.

The error message never echoes server paths.

---

## 4. Model weights and remote code

Model loading uses `trust_remote_code=True`, and checkpoints are fetched from Hugging Face by repository name without a pinned revision. A compromised or replaced model repository can therefore execute code on your machine at load time.

- Load only the official ACE-Step checkpoints and the LoRA / LoKR adapters you trained yourself.
- Do not point `ACESTEP_CONFIG_PATH`, `ACESTEP_LM_MODEL_PATH` or `/v1/lora/load` at files from an untrusted source.
- Training resume files are loaded with `torch.load(weights_only=True)`. A resume file that needs arbitrary unpickling is refused rather than executed.
- Prefer `.safetensors` over pickle-based formats when you have a choice.

---

## 5. Dependencies

Install from the lock file so you get the audited versions rather than whatever resolves on the day:

```bash
uv sync
```

Known gap: `pillow` is held below 13.0 by `gradio`, and pillow 12.x still carries advisories that are fixed only in 13.0. If you do not need the Gradio UI in a deployment, you can override the pin in your own environment; otherwise keep the API off the network (sections 1 and 2) so untrusted image input cannot reach it.

---

## 6. Docker

The container runs as root and binds `0.0.0.0` inside the container. Recommended run flags:

```bash
docker run --gpus all \
  -p 127.0.0.1:8001:8001 \
  -e ACESTEP_API_KEY="<your key>" \
  -e ACESTEP_ALLOWED_PATHS=/data \
  -v /path/to/data:/data \
  ace-step:latest
```

Publishing to `127.0.0.1` keeps the port off the LAN while the container keeps its internal bind. Mount only the directories the training routes need.

---

## 7. Checklist

- [ ] Server bound to `127.0.0.1`, or behind a TLS reverse proxy with source restrictions
- [ ] `ACESTEP_API_KEY` set to a random value of at least 32 bytes
- [ ] Server started from the project directory; `ACESTEP_ALLOWED_PATHS` lists only the data directories you need
- [ ] Only official checkpoints and your own adapters are loaded
- [ ] Installed from `uv.lock`
- [ ] Docker port published to loopback, data mounted read-only where possible

---

## Environment variable reference

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `ACESTEP_API_HOST` | `127.0.0.1` | Bind address for the REST API and the TensorBoard it launches |
| `ACESTEP_API_PORT` | `8001` | REST API port |
| `ACESTEP_API_KEY` | (empty) | Bearer key for every REST and OpenRouter-compatible route; empty disables auth |
| `ACESTEP_ALLOWED_PATHS` | (empty) | Extra directories the training, dataset and LoRA routes may read or write |
| `ACESTEP_TMPDIR` | `.cache/acestep/tmp` | Temp directory; also an allowed root |
| `OPENROUTER_API_KEY` | (empty) | Bearer key for the standalone OpenRouter server only |

Full variable list: [API Reference, section 13](./API.md#13-environment-variables).

---

## Reporting a vulnerability

See [SECURITY.md](../../SECURITY.md). Do not open a public issue for a security problem.
