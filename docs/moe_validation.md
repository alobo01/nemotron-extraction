# Matched-precision MoE replay

This is an isolated operator experiment, not a production full-model parity claim.
The reference implementation is the three NVIDIA classes selected by AST from
the pinned checkpoint source (`SOURCE_SHA` in the harness). Their expert weights
share verified native vLLM tensors. No full-model Transformers/Mamba import is
needed inside vLLM's runtime.

The same configuration is applied to both operator paths:

- Identical checkpoint values and saved normalized inputs for same-input tests.
- IEEE FP32 PyTorch backends; reduced BF16 reductions disabled.
- FP32 gate projection and sigmoid, selection-only correction bias, six of 128
  experts, normalized selected weights, routed scale 2.5, ReLU-squared experts.
- `TRITON_F32_DEFAULT=ieee` set before starting either runtime. This controls
  default Triton dot precision; explicit kernel precision may still differ.
- Separate BF16 and FP32 expert profiles. FP32 widens the BF16 checkpoint;
  it does not recover information absent from those weights.
- Both native selector and explicit common-selector controls are measured.
  The latter overrides vLLM selection using the frozen PyTorch top-k routine,
  retaining native vLLM expert kernels. Raw selector disagreements are saved,
  not treated as a reason to discard a case.

An immutable run includes `reference_model.py`, the harness, `benchlib`, source
checksums, a storage preflight, a fixture, saved inputs, per-case output tensors,
per-case receipts, `result.json`, and unique Slurm logs. The tiny unused vocabulary
allows the original operator dimensions and weights to fit one 24-48 GB GPU.

Create a fresh run root with a `cases/` directory. Copy the pinned checkpoint's
`modeling_nemotron_h.py` to `reference_model.py` and verify its expected checksum.
Use the established cluster containers/environment and offline checkpoint cache:

```bash
export TRITON_F32_DEFAULT=ieee
# HF environment builds the ME fixture from original layer 0/1 weights.
python benchmarks/moe_attribution.py prepare --root "$RUN" --source "$MODEL" \
  --traces "$HF_TRACE" "$VLLM_TRACE"
# vLLM environment runs both the frozen PyTorch equations and native operators.
python benchmarks/moe_attribution.py run --root "$RUN" --source "$MODEL"
python benchmarks/moe_attribution.py verify --root "$RUN"
```

The first and last commands require Torch/safetensors and the extraction package;
preparation additionally requires the compatible Transformers/Mamba environment.
The reference model source and original checkpoint remain read-only. For worker
extension imports, put the directory containing `moe_attribution.py` and
`benchlib/` on `PYTHONPATH` in both the driver and worker.

Counterfactuals keep the candidate input and expert arithmetic fixed, replace
its chosen IDs with the reference set, and recompute routing weights from the
candidate logits. Native vLLM includes a sham intervention with its own IDs and
the same recomputation. Report the raw-vs-sham difference separately. Error
reduction includes interactions with inherited activation errors; it is not an
additive fraction of all full-model error. Slot permutations do not count as
expert-set changes. A small cutoff gap is not an exact tie.

The checked profile is in [`matched_precision.json`](matched_precision.json).
The Triton default is documented in the [official Triton repository](https://github.com/triton-lang/triton#debugging-tips).
PyTorch controls are version-aware (`benchlib.precision.ieee_precision`) and are
restored after the operator experiment. They are process-wide: apply them inside
an isolated validation worker. Setting `dtype` alone does not configure the
native router override or guarantee matching kernel arithmetic.


FP32 on small GPUs additionally needs the private tile recorded in the profile.
The default tile exceeded the RTX A6000 shared-memory limit. Generate the file
inside the allocated GPU environment before starting vLLM:

```bash
export VLLM_TUNED_CONFIG_FOLDER="$RUN/moe-configs"
python - <<'PY'
import json, os
from pathlib import Path
import torch
folder = Path(os.environ["VLLM_TUNED_CONFIG_FOLDER"])
folder.mkdir(parents=True, exist_ok=False)
profile = json.loads(Path("docs/matched_precision.json").read_text())
gpu = torch.cuda.get_device_name(0).replace(" ", "_")
name = f"E=128,N=1856,device_name={gpu},dtype=float32.json"
(folder / name).write_text(json.dumps(profile["native_fp32_tile"]["contents"]))
PY
```

The follow-up checks that vLLM loads this exact configuration before engine
startup. This is a resource-safe tile for accuracy validation; no throughput
optimality is claimed. See the [official vLLM tuning interface](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py).
Use `run --profiles float32` to run only that profile in a fresh root with
explicitly staged completed BF16 cases. The harness refuses to overwrite case
directories and records reused case names. Final verification still requires
all eight expected cases and checks their saved tensors independently.


Verified results (21 October 2026): job **3640851** completed **0:0** in
**4m53s** on RTX A6000, `serv-9221`, via RTXA6000-MLT. Four preserved BF16
cases from 3640646 and four new FP32 cases passed final eight-case verification.
All sixteen shared-selector same-input comparisons meet the unchanged strict
gates. Full-anchor maximum relative L2: BF16 **0.00365224**, FP32 **1.05279e-6**.
Both profiles used the same GPU model/node and frozen inputs/weight values.

| Native BF16 layer | Changed sets / 2048 | Single swaps | Bad cosine < .99 before -> after forcing reference IDs | Squared error removed |
|---|---:|---:|---:|---:|
| 1 | 39 | 39 | 4 -> 0 | 64.6% |
| 27 | 356 | 342 | 276 -> 73 | 40.5% |
| 51 | 261 | 256 | 7 -> 3 | 17.4% |

Of the recovered bad rows, 4/193/3 involve exactly one expert changing.
For a corrected sixth/seventh score gap <= 0.001, recovered counts are 3/23/0.
The table measures controlled interventions, not a causal fraction of every
full-model discrepancy. Reference-only layer-27 replay has 277 -> 74 bad rows;
the native controlled replay has 276 -> 73. Both recover 203; use the appropriate
denominator. No new .99-bad rows appear in these interventions.

FP32 expert execution leaves the 39/356/261 cross-input route changes intact:
the frozen upstream activations still differ. Each profile also has one native
layer-51 same-input selector disagreement at an exact cutoff tie, despite exact
logits. The explicitly shared selector removes it. FP32 projection alone does
not guarantee identical tie resolution. CPU centering of the large common
correction-bias offsets removes observed CPU ties and matches FP64 selections;
this alternative has not been validated in native GPU execution.

The original full-model pilot remains at 3/18 passing sites. Production arithmetic
defaults have not changed. Full-model validation of the matched profile remains
necessary; widening BF16 checkpoint values does not recover missing precision.
The first two diagnostic jobs remain preserved as failed (route-identity assertion,
then oversized FP32 tile); their successful BF16 artifacts were retained and
verified again. No jobs were cancelled or failed artifacts overwritten.

Machine-readable evidence: [`moe_validation_results.json`](moe_validation_results.json),
[`mamba_validation_results.json`](mamba_validation_results.json),
[`l51_router_precision.json`](l51_router_precision.json), and
[`router_centering_cpu.json`](router_centering_cpu.json). The MoE receipt includes
all per-case counters/gates, software versions, source/anchor hashes, original
weight checks, tile-loader receipt and reused-case provenance.
