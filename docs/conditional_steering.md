# Conditional steering: a vehicle gate × a damage writer

## Why global steering affects unrelated objects

Ordinary steering adds the same residual-space vector to every request:

```
h' = h + α · v_damage
```

`v_damage` comes from `mean(damaged − intact)`, decoded through the SAE (see
`research/vector_builders/`). It says **what** to write. Nothing in the formula says
**when** to write it. If the vector is a fairly general "physical damage" direction,
it will make a chair or a tennis racket look damaged as readily as a vehicle. That
says nothing bad about the vector. The intervention is simply input-agnostic, and
adding more non-vehicle images to the A/B contrast cannot change that: a constant
vector is still constant.

## Detector × writer

Conditional steering adds an explicit READ step:

```
h' = h + α · g(h) · v_damage
```

| Part | Question | How it is obtained | Stored as |
| --- | --- | --- | --- |
| **Writer** `v_damage` | Which way does the residual move when the model represents damage? | Existing A/B profile → vector builder → SAE decoder | `steering_vectors.safetensors` |
| **Detector** `w`, `b`, `τ` | Is the target object in the image? | Separate labelled gate manifests → logistic-regression probe | `conditional_gate.safetensors` + config |

The gate is computed from **visual tokens only**. At gate layer `l_g` (default:
layer 9, the earliest dashboard layer):

```
z = mean of h_t at layer l_g over the image-token positions t   (Gemma image_token_id)
s = w · ((z − center) / scale) + b
hard (default): g = 1 if s ≥ τ, otherwise 0
soft (research): g = sigmoid((s − τ) / T)
```

Text tokens never enter `z`, so the literal word "car" or "vehicle" in the question
cannot open the gate. In this repository's chat layout the image comes before the
question, and Gemma 3 uses causal attention outside the image block, so the question
cannot change the image-token states either. A prompt with no image tokens gets
`g = 0`. In hard mode a closed gate means the hooks return the hidden state
unchanged: exactly zero intervention, not a small one.

Both directions are *operationally identified*. Neither the probe weight nor any SAE
latent is shown to literally "be" vehicle or damage. Conditional steering tests a
stronger claim than ordinary steering: that one internal reading can causally decide
whether another direction is written.

## Runtime semantics

The implementation is in `sae_dashboard/model_runtime.py`
(`register_gated_steering_hooks`) and `research/conditional_steering.py`.

- **Prefill.** On its first call, the gate-layer hook pools the *unmodified*
  layer output over the image tokens and stores `score`, `g` and the number of image
  tokens in a request-local `GateTrace`. Steering at that same layer is applied after
  this read.
- **Decoding.** Cached decoding steps have no image tokens. They reuse the stored `g`.
- **Layer order.** Any layer with nonzero strength must be at or after the gate layer.
  Otherwise the call is rejected before any hook is installed. The gate layer does not
  need to be a steering layer.
- **Isolation.** There is no global gate variable. Each request creates its own
  `GateTrace`. All model calls hold `MODEL_LOCK`, and hooks are removed in `finally`,
  including after exceptions.
- **Legacy behaviour.** With `gate=None` (gate disabled), the original hook code runs
  unchanged. With `g = 1`, the added delta equals `α·v` exactly, so the results match
  global steering.

The coherence benchmarks and the Extra tab still use global steering. Conditional
steering is available in **4 · Base vs. steered**, in leakage evaluation and in exports.

## Building and enabling the gate

1. Build the damage profile and vectors as usual from
   `data/physical_damage/manifest.json` (sections 1–3).
2. Prepare gate manifests. See [data/vehicle_gate/README.md](../data/vehicle_gate/README.md)
   for the format, target-category choice and dataset requirements. **No real gate
   dataset ships with the repository.**
3. In **4 · Base vs. steered → Conditional steering · vehicle gate**:
   - load a training manifest (`"split": "train"`) and a validation manifest
     (`"split": "validation"`);
   - choose the gate layer, mode, maximum validation false-positive rate (default 0)
     and L2 penalty;
   - click **Build vehicle gate**. The status line shows target, layer, threshold and
     validation TPR/FPR/TNR;
   - tick **Enable vehicle gate**.
4. Run **Run base + steered**. A line under the answers reports the gate score,
   threshold, `g`, and whether steering was applied.
5. **6 · Export VLM** includes the gate whenever it is enabled.

Invalidation rules:

- Rebuilding the A/B profile or vectors keeps a valid gate.
- Loading new gate manifests, or changing the gate layer, mode, temperature, FPR
  budget or L2, discards the gate. A rebuilt gate starts disabled.

### Calibration procedure

`research/conditional_steering.fit_gate`:

1. Capture pooled image-token residuals at the gate layer for every example, with no
   steering.
2. Center the features using the training mean, and divide them by one scalar `scale`
   (mean centered norm / √width). `center` and `scale` are stored and reapplied exactly
   at inference.
3. Fit class-balanced L2 logistic regression on the **training** split only, using
   full-batch float64 LBFGS from a zero initialization. The result is deterministic.
4. Choose `τ` on the **validation** split only. Candidates are midpoints between
   consecutive validation scores. The chosen `τ` has the highest TPR subject to
   `FPR ≤ budget`. The default budget is 0: no validation non-target may open the gate.
5. Store the layer, weight, bias, threshold, center, scale, target, mode, temperature,
   and the training and validation confusion matrices.

## Evaluating leakage

**Leakage evaluation**, under the gate accordion, takes a labelled gate manifest
(ideally `"split": "test"`, disjoint from calibration). For each image it measures the
next-token log-odds of *Yes* versus *No* for two questions:

- `damaged?` — "Is the main object in this image physically damaged? Answer Yes or No."
- `intact?` — "Is the main object in this image intact and undamaged? Answer Yes or No."

Each question is run three times: base, global steering and gated steering, all with
the strengths from the sliders. Then:

```
Δ_q          = log-odds_steered(q) − log-odds_base(q)
damage_shift = Δ_damaged? − Δ_intact?
E_target     = mean(damage_shift | target)
E_leak       = mean(|damage_shift| | non-target)
selectivity  = E_target / (E_leak + ε)
```

**DAMAGE versus YES.** A vector that only biases the model toward the token "Yes"
raises both log-odds, so it cancels out of `damage_shift`. Damage semantics should
raise the first log-odds and lower the second. You can also tick **Also generate
open-ended descriptions** to record base, global and gated answers to "Describe the
main visible object and its physical condition." for manual review.

The report also includes:

- gate TPR, FPR and TNR on the evaluation set;
- false-positive rate on hard negatives alone;
- `closed_gate_max_abs_delta`, which must be exactly 0 in hard mode.

When experiment saving is on, every record is saved: prompts, gate scores, all three
log-odds and the optional texts.

Controls to run:

| Case | Expected |
| --- | --- |
| Intact target vehicle, gated | Gate open; `damage_shift` comparable to global steering |
| Already damaged target vehicle | Gate open; values remain finite and the output stays coherent |
| Tennis racket, chair, cat, person, building | Gate closed; gated Δ = 0; answer identical to base |
| Hard negatives (car or military vehicle, depending on target; bus, truck, bicycle, wheel, parking lot) | Matches the declared target definition; report hard-negative FPR |
| Gate disabled | Reproduces global steering, including leakage |
| Random or nonmatching vectors (Extra tab) | Unchanged |

## Portable export

When the gate is enabled, the export uses `"format_version": 2` and adds
`conditional_gate.safetensors` (`weight`, `center`) plus a `conditional_gate` section
in `steering_config.json` (layer, bias, threshold, scale, mode, temperature, target,
calibration metadata). The bundled `steered_model.py` contains its own copy of the gate
math (unit tests check it against the research module). It computes `g` from Gemma's
image tokens during prefill and needs no SAE.

The loader rejects:

- unknown format versions, or a v1/v2 export that contradicts its gate section;
- tensor shapes that do not match the hidden width;
- NaN or infinite values;
- an unknown mode, a non-positive scale or temperature, or a non-numeric threshold;
- gate layers outside the model;
- nonzero steering layers before the gate layer.

Ungated exports remain `format_version: 1` and load as before.
`vlm.generate(prompt, image, return_gate=True)` returns `(text, {score, value, image_tokens})`.

## Limitations

- **Image-level only.** The gate says the target is somewhere in the image. With a
  vehicle, a person and a tree in one scene, the damage direction is still written into
  the last-token or all-position residual, not bound to the vehicle's tokens. A
  per-token gate `g_t` is a natural extension, since `pool_visual_tokens` and the
  hooks already work on image masks.
- **Batch size 1.** The runtime matches the dashboard.
- **Calibration quality bounds everything.** With few or homogeneous images (for
  example, all on one studio background), the probe will find shortcuts. The
  validation metrics are measured on your images; they are not general guarantees.
- **Damage data domain.** The current damage vector comes from military vehicles. A
  "car" gate composed with it relies on the damage direction transferring.
- **Soft mode** lets weakly positive non-targets receive partial steering. Use hard
  mode when preventing leakage is the goal.

## Software tests versus behavioural evidence

`make test` (`tests/test_conditional_steering.py`) uses fake layers and synthetic
tensors. It checks:

- gate math and visual-token pooling;
- that text tokens cannot move the gate;
- `g=1 ⇒ +αv` and `g=0 ⇒ exactly unchanged`;
- that the gate is computed in prefill and reused while decoding;
- read-before-write at a shared layer;
- layer-order rejection and hook cleanup after errors;
- per-request trace isolation;
- session invalidation semantics;
- export round trips and malformed-artifact rejection.

These tests show that the mechanism is implemented as specified. They say **nothing**
about whether Gemma's visual residuals separate vehicles from non-vehicles, or whether
gated damage steering is selective. That requires the leakage evaluation on real
images with real Gemma weights.
