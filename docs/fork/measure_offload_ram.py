"""Run N generations with DiT offload on and print process private memory after each.

Usage: python measure_offload_ram.py <label> <n_gens> <save_dir>
Run from the repo checkout whose acestep package should be measured.
"""
import gc
import os
import sys
import time

import psutil

os.environ.setdefault("ACESTEP_OFFLOAD_DIT_TO_CPU", "true")

label, n_gens, save_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cwd = os.getcwd()
sys.path.insert(0, cwd)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
import acestep  # noqa: E402

proc = psutil.Process()


def priv_gb() -> float:
    gc.collect()
    return proc.memory_info().private / 1024**3


print(f"[measure] label={label} acestep={acestep.__file__}", flush=True)
print(f"[measure] private before init: {priv_gb():.2f} GB", flush=True)

# Checkpoints live in the main checkout; project_root only resolves paths.
project_root = "W:/dev/ACE-Step-1.5"
dit = AceStepHandler()
status, ok = dit.initialize_service(
    project_root=project_root,
    config_path="acestep-v15-turbo",
    device="cuda",
    use_flash_attention=False,
    compile_model=False,
    offload_to_cpu=True,
    offload_dit_to_cpu=True,
    quantization=None,
)
assert ok, status
llm = LLMHandler()
status, ok = llm.initialize(
    checkpoint_dir=os.path.join(project_root, "checkpoints"),
    lm_model_path="acestep-5Hz-lm-1.7B",
    backend="pt",
    device="cuda",
    offload_to_cpu=False,
    dtype=None,
)
assert ok, status
print(f"[measure] private after init: {priv_gb():.2f} GB", flush=True)

params = GenerationParams(
    caption="Upbeat indie pop with bright guitars and a steady drum groove",
    lyrics="[Verse]\nMorning light across the floor\nWe open up the door\n[Chorus]\nRun with me tonight\nUnder city light",
    vocal_language="en",
    bpm=120,
    duration=60,
    inference_steps=8,
    seed=1234,
    thinking=True,
)
config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[1234], audio_format="wav")

readings = []
for i in range(1, n_gens + 1):
    t0 = time.time()
    result = generate_music(dit, llm, params, config, save_dir=save_dir)
    dt = time.time() - t0
    gb = priv_gb()
    readings.append(gb)
    print(f"[measure] gen {i}: success={result.success} {dt:.0f}s private={gb:.2f} GB", flush=True)

print(f"[measure] RESULT {label}: " + " -> ".join(f"{g:.1f}" for g in readings) + " GB", flush=True)
