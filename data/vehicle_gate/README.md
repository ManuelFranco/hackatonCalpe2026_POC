# Vehicle gate calibration data

This folder defines the manifest format for the **conditional-steering gate**: the
READ half of `h' = h + α·g(h)·v_damage`. The gate's question is "is the target object
in the image?" The damage vector's question is "which way does the model move when the
object is damaged?" These are separate questions, so the gate gets separate data.
Loading gate manifests never changes the A/B damage profile or its vectors.

**No real calibration set ships with the repository.** `gate.template.json` is a
format template (`"template": true`). The loader rejects it, and the dashboard dropdown
hides it. The repository currently has no suitable non-target photographs. `data/colors`
holds only synthetic letters, which would make every negative trivially easy. Any gate
built from it would be a shortcut detector.

## Target category: be explicit

Every `data/physical_damage` image is a **military ground vehicle** (tanks, armored
personnel carriers, armored trucks), photographed against the same concrete
studio wall. None of them are civilian cars. The template therefore uses
`"target": "military_ground_vehicle"`.

To gate on **cars** instead, set `"target": "car"`, use car positives, and list
military vehicles as hard negatives. The damage vector is then still estimated from
military vehicles, which is a transfer assumption that you need to test. The code
never broadens or renames the target. Training and validation manifests must declare
the same `target`.

## Manifest format

```json
{
  "version": 1,
  "kind": "binary_gate",
  "name": "Vehicle gate · train",
  "target": "military_ground_vehicle",
  "split": "train",
  "asset_root": "vehicle_gate",
  "prompt": "Describe the image.",
  "examples": [
    {"id": "tank_001", "label": "target", "category": "tank", "image": "images/target/tank_001.png"},
    {"id": "bus_001", "label": "non_target", "category": "bus", "hard_negative": true, "image": "images/hard_negative/bus_001.png"}
  ]
}
```

- `kind` must be `binary_gate`. Gate filenames should not contain `manifest`, so they
  stay out of the A/B manifest dropdown.
- `split` is `train`, `validation` or `test`. The probe weights are fitted on
  `train`. The threshold is chosen on `validation` only. Use `test` for the leakage
  evaluation.
- `label` is `target`/`positive`/`1` or `non_target`/`negative`/`0`.
- Every example needs an `image`, because the gate reads only image-token states. The
  `prompt` is only there to build a normal chat input. In Gemma 3 the image tokens
  come before the question and use causal attention, so text after the image cannot
  change them.
- Paths resolve inside `GEMMA_DATA_ROOT`, using the same rules as A/B manifests.

The loader enforces a few rules:
- The same image cannot appear in two gate splits.
- Gate training and validation sets cannot reuse any image from a repository A/B
  manifest marked `"split": "validation"` or `"split": "test"`, such as the
  physical-damage holdout.

## What a useful calibration set contains

| Group | Examples | Notes |
| --- | --- | --- |
| Targets | tanks, APCs, armored trucks, armored cars, with and without damage | Many viewpoints, backgrounds, lighting conditions and occlusion levels. Include damaged targets, because the gate must still open on them. |
| Hard negatives | civilian cars, buses, trucks, vans, tractors, construction machinery, motorcycles, bicycles, wheels, roads, parking lots | Mark these with `hard_negative: true`. Their false-positive rate is reported separately. |
| Same-setting negatives | chairs, rackets, boxes and so on, photographed in the **same concrete studio** | Without these, the probe can learn the background instead of the object. |
| Easy negatives | tennis racket, chair, cat, dog, person, building, tree | Useful, but not enough on their own. |

Aim for at least tens of examples per group in each split. Gemma 3 4B's residual
width is 2560, so with very few examples the probe can fit anything, and the
validation metrics are the only honest signal.
