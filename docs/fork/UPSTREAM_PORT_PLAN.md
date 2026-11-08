# Upstream triage and port plan

Fork: `owenpkent/ACE-Step-1.5`. Upstream: `ace-step/ACE-Step-1.5`.
Triage date: 2026-11-07, against upstream `main` at `ca1e85f`.

This is a working document for the fork. It records which open upstream pull requests
and issues are worth bringing into the fork, and the order to do it in.

Status 2026-11-07: Phases 0 to 3 are done and every fork PR below is merged into `main`.
The per-phase worktrees have been removed. Phase 4 is still open.

## Summary

- Upstream had 67 open PRs and 95 open issues. Most are stale support questions,
  feature requests, drafts, or large refactors.
- Six upstream PRs are worth cherry-picking. None is approved upstream yet, and all
  were reported mergeable by GitHub. None of them touch the same file, so they can be
  ported independently. Verified 2026-11-07: every PR head is a plain commit series with no
  merges, and each merges cleanly onto `ca1e85f` (`git merge-tree`).
- Three fixes are worth writing in the fork ourselves, one of them security related.
- Two upstream PRs collide with the fork's security series and should be skipped or
  sequenced after it.

## Fork security series (merged)

| Fork PR | Branch | Topic |
|---|---|---|
| 1 | `fix/api-auth-openrouter-routes` | API key on OpenRouter routes, constant-time key compare |
| 2 | `fix/network-exposure-defaults` | Loopback binding for launchers and TensorBoard |
| 3 | `fix/safe-torch-load-resume` | `weights_only=True` on training resume |
| 4 | `ci/pin-actions-to-sha` | Actions pinned to SHAs, read-only token |
| 5 | `chore/bump-vulnerable-deps` | Dependency bumps past known advisories |
| 6 | `fix/confine-training-route-paths` | Training and dataset paths confined to allowed roots |
| 7 | `docs/security-hardening` | Operator security hardening guide |

Merged alongside, outside this plan: 19 (transitive dependency bumps), 20 (scanner-found
crashes), 21 (Docker non-root user), 22 (generic error details on the Gradio API routes),
29 (startup warning for broad allowed roots), 30 (path policy: outside links and case
variants), and Dependabot 23 to 28 (GitHub Actions bumps).

## Upstream PRs to port

Ranked by value to the fork. Head SHAs are the PR heads at triage time.

| Rank | PR | Fixes | Size | Risk | Upstream review | Head |
|---|---|---|---|---|---|---|
| 1 | [#1368](https://github.com/ace-step/ACE-Step-1.5/pull/1368) | Windows RAM grows 4 to 12 GB per generation with DiT offload, then crashes | +198/-10, 4 files | Medium | Review required | `2422ec4` |
| 2 | [#1352](https://github.com/ace-step/ACE-Step-1.5/pull/1352) | Every song with lyrics fails with NaN latents on float16-only GPUs (Pascal, Volta, Turing) | +327/-0, 4 files | Low-medium | Review required | `c751457` |
| 3 | [#1357](https://github.com/ace-step/ACE-Step-1.5/pull/1357) | Metadata records different seeds than were used, omits batch size (issue #1347) | +126/-18, 3 files | Low | Review required | `fb91f5b` |
| 4 | [#1309](https://github.com/ace-step/ACE-Step-1.5/pull/1309) | Edit mode device-mismatch crash when DiT offload is on | +97/-22, 2 files | Low | Changes requested | `86cf3a2` |
| 5 | [#1321](https://github.com/ace-step/ACE-Step-1.5/pull/1321) | Missing `logger` import turns a logged warning into `NameError` | +2/-0, 1 file | None | Review required | `33e2d60` |
| 6 | [#1188](https://github.com/ace-step/ACE-Step-1.5/pull/1188) | MP3 export fails under `uv run` on Windows when ffmpeg is not on PATH | +45/-2, 1 file | Low | Review required | `0e3cab6` |

Files touched, to confirm there is no overlap between ports:

| PR | Files |
|---|---|
| #1368 | `acestep/core/generation/handler/generate_music_execute.py`, `service_thread.py`, and their tests |
| #1352 | `acestep/core/generation/handler/fp16_lyric_encoder.py` (new), `init_service_loader.py`, tests |
| #1357 | `acestep/inference.py`, two new tests |
| #1309 | `acestep/core/generation/handler/service_generate_flow_edit.py` and its test |
| #1321 | `acestep/acestep_v15_pipeline.py` |
| #1188 | `acestep/audio_utils.py` |

Notes:

- **#1309 versus #1331.** Both fix the same Edit mode crash in the same file. #1331 is
  newer (2026-09-17) and has no review block, but it ships no test. Take #1309 because it
  adds a regression test. The maintainer blocked it on one stale test,
  `CoverStillRunsLmTests.test_no_edit_in_skip_lm_tasks`, which still expects an old literal
  set of task names. Fix that test when porting. Do not also take #1331.
- **#1368** changes the generation thread model. Port it on its own branch and run a
  few generations with DiT offload on Windows before merging, watching private memory.
- **#1352** adds a module and a loader hook only on float16 GPUs. Confirm with a unit run
  that the bfloat16 path is untouched.

## Fixes to write in the fork

| Rank | Source | Problem | Confirmed on main | Size |
|---|---|---|---|---|
| 1 | Upstream PR [#1130](https://github.com/ace-step/ACE-Step-1.5/pull/1130) (stalled) | API key accepted from the request body as `ai_token`, where it can reach logs and proxies | Yes, in two places | Small |
| 2 | Issue [#615](https://github.com/ace-step/ACE-Step-1.5/issues/615) | LoKR adapter loads but has no effect; key mismatches are silent | Yes | Small |
| 3 | Issue [#1370](https://github.com/ace-step/ACE-Step-1.5/issues/1370) | Full 3.76 GB re-download when only the LM folder is missing | Yes | Small-medium |

### 1. Remove the `ai_token` body credential

- The body credential is read in `acestep/api/http/auth.py` (around line 42) and again in
  `acestep/ui/gradio/api/api_routes.py` (around line 62). Upstream #1130 only fixes the
  first file.
- Upstream #1130 has been stalled since 2026-08-28. The maintainer agreed with the
  direction but blocked it because `acestep.api.http.auth_test` was not updated.
- Fork PR 1 already rewrites both auth functions for constant-time comparison. Branch
  this fix from `fix/api-auth-openrouter-routes`, not from `main`, to avoid conflicts.
- `ai_token` is documented in `docs/en/API.md` and its `zh`, `ja`, and `ko` translations.
  Removing it breaks any client that sends the key in the body.
- **Decision (taken, see Phase 3):** remove outright, or reject by default with an opt-in environment
  flag for old clients. Recommended: opt-in flag, off by default, with a warning in the
  log when it is used. This keeps the public interface available, as `AGENTS.md` asks.
- Tests: header auth still works; body token rejected by default; body token accepted
  only with the flag; same three cases for the Gradio routes.

### 2. Surface LoKR key mismatches (issue #615)

- `acestep/core/generation/handler/lora/lifecycle.py` (around line 181) calls
  `lycoris_net.load_weights(weights_path)` and discards the result.
- Fix: inspect the returned missing and unexpected keys. Log a warning naming the counts,
  and raise if no keys matched, since the adapter would then do nothing.
- Tests: a mocked `load_weights` returning all keys matched, partly matched, and none
  matched.

### 3. Skip the LM check when the LM is not requested (issue #1370)

- `_ensure_models_present` in
  `acestep/core/generation/handler/init_service_downloads.py` (around line 31) calls
  `check_main_model_exists()`, which requires every component including the LM.
- Fix: when `init_llm` is off, check only the components that will be loaded. When only
  the LM is missing, download only the LM. `_ensure_models_present` does not receive the
  LM flag today; plumb it from the caller in `init_service_orchestrator.py` (around line
  138). `model_downloader.ensure_lm_model` already exists, so no new download code is needed.
- Tests: LM missing with `init_llm` off makes no download; LM missing with `init_llm` on
  downloads the LM only; DiT missing still triggers a download.

## Upstream items to skip or sequence

- **#1131, auth fails open when no key is set.** Edits the same auth file as fork PR 1
  and has changes requested. Skip it. If fail-closed auth is wanted, write it on top of
  fork PR 1.
- **#1278, SBOM and provenance attestations.** Harmless, but conflicts textually with
  fork PR 4 in `.github/workflows/container.yml`. If wanted, apply it after PR 4 merges and
  pin its action to a commit SHA.
- **Issue #614, "Rejected unsafe directory path".** Not a bug to port, but a warning.
  Fork PR 6 tightens the same check, so expect more of these reports. Users keep their
  music outside the install folder. Consider a clearer error message that names the
  allowed roots, and a configurable extra root.

Borderline items not taken, for the record:

- #1302 keeps the requested vocal language over the LM's detection. Useful, but changes
  were requested; revisit if upstream lands it.
- #1369 fixes a Windows RSS readout that always shows 0 MB. Log-only; port with #1368
  if convenient.
- #1283 same-seed reproducibility and #1311 LM engine leak on reinit. Both have changes
  requested, and #1311 is partly superseded by merged #1310.
- #1262 to #1264 multi-GPU series, #1301 and #1129 GPU-config refactors. Too large.
- Issue #662, Opus and AAC export fails without the torchaudio ffmpeg backend. Real, but
  several upstream PRs overlap; wait for one to settle.
- Issue #658, ROCm launchers force `HSA_OVERRIDE_GFX_VERSION=11.0.0`. Real, but ROCm
  only and outside the fork's hardware.

## Plan

### Phase 0: setup

Already done in this clone: the `upstream` remote exists and PR heads are fetched under
`refs/upstream-pr/<n>`. For a fresh clone:

```bash
git remote add upstream https://github.com/ace-step/ACE-Step-1.5.git
git fetch upstream main
```

### Phase 1: low-risk ports

One branch and one fork PR per upstream PR, so each can be dropped if upstream changes
it. Fetching the PR head and cherry-picking keeps the original author on each commit.

```bash
git fetch upstream pull/1321/head:upstream-pr-1321
git switch -c port/1321-pipeline-logger main
git cherry-pick main..upstream-pr-1321
uv run python -m unittest discover -s . -p "*_test.py"
uv run python -m unittest discover -s . -p "test_*.py"
```

Order: #1321, #1357, #1188, #1352, #1309 (with its stale test fixed).

Status 2026-11-07: done and merged.

| Upstream | Fork PR | Branch | Notes |
|---|---|---|---|
| #1321 | 8 | `port/1321-pipeline-logger` | |
| #1357 | 9 | `port/1357-seed-metadata` | |
| #1188 | 10 | `port/1188-ffmpeg-uv-windows` | AI co-author trailer dropped from the upstream commit |
| #1352 | 11 | `port/1352-fp16-lyric-encoder` | AI co-author trailer dropped; one pre-existing `init_service_test` error is identical on `main` |
| #1309 | 12 | `port/1309-edit-offload-device` | Upstream already updated the stale test; fork commit adds `encoding="utf-8"` because the test reads `inference.py` and fails on Windows (cp1252) on `main` too |


### Phase 2: the Windows memory fix

Port #1368 on `port/1368-windows-offload-ram`, together with #1369 if wanted. Before
opening the fork PR, run five generations with DiT offload on and record private memory
after each. It should stay flat.

Status 2026-11-07: done and merged, fork PR 13. AI co-author trailers dropped from both upstream
commits. Measured with `docs/fork/measure_offload_ram.py` (run from the checkout under
test; checkpoints read from the main checkout). RTX 5090, 96 GB RAM, torch 2.7.1+cu128,
`acestep-v15-turbo` + `acestep-5Hz-lm-1.7B` (pt), five 60 s generations, private bytes:

| Build | Gen 1 | Gen 2 | Gen 3 | Gen 4 | Gen 5 |
|---|---|---|---|---|---|
| `main` | 20.2 GB | 23.4 GB | 26.6 GB | 29.7 GB | 32.9 GB |
| port | 20.2 GB | 20.3 GB | 20.3 GB | 20.3 GB | 20.3 GB |

Gotcha: `offload_dit_to_cpu` alone does nothing; `_load_model_context` returns early
unless `offload_to_cpu` is also on. A first run with only the DiT flag showed no growth
on either build. #1369 ported as fork PR 14 (`port/1369-windows-rss-readout`): `_get_rss_mb()` returns 0 on `main` and the real figure on the port.

### Phase 3: fork-authored fixes

1. `fix/remove-body-api-token`, branched from `fix/api-auth-openrouter-routes`. Became
   fork PR 18.
2. `fix/lokr-key-mismatch-warning` from `main`.
3. `fix/skip-lm-check-without-llm` from `main`.

Status 2026-11-07: done and merged.

| Fix | Fork PR | Notes |
|---|---|---|
| LoKR key mismatch | 16 | LyCORIS `load_weights` returns a dict (`"missing keys"`, `"unexpected keys"`), not torch `_IncompatibleKeys`. New `lora/lokr_load_check.py`; `add_lora` already converts the raise into the `❌` message |
| LM-aware download | 17 | New `download_main_component` fetches one folder via `_smart_download(allow_patterns=...)`, verified live against HuggingFace. `initialize_service(require_lm=True)` added; API paths pass it, Gradio pipeline left at default |
| Body `ai_token` | 18, base `fix/api-auth-openrouter-routes` | Decision taken: opt-in `ACESTEP_ALLOW_BODY_API_TOKEN`, off by default; body token ignored with a warning and the header check runs as before. Retargeted to `main` after fork PR 1 merged |

### Phase 4: upstream follow-up

- Once the fork PRs merge, consider offering the fork-authored fixes upstream. Comment on
  #1130 and on issues #615 and #1370 as a separate, deliberate step.
- Re-run this triage monthly. Drop any port whose upstream PR merges, and rebase onto
  the merged version instead.

## Done when

- Each port has its own fork PR with the upstream PR linked in the description.
- The focused tests for each touched module pass.
- #1368 has a recorded memory measurement on Windows.
- The `ai_token` decision is made and recorded in fork PR 18. (Done: opt-in flag, see Phase 3.)
