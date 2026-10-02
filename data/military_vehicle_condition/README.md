# Military Vehicle Condition — Image Pairs

One use case with 20 numbered A/B image slots, following the paired CWE dataset convention.

- **A · Not destroyed:** a vehicle that is not destroyed; this label does not imply operational readiness.
- **B · Destroyed:** a destroyed vehicle.
- Pair `01` matches A `01` with B `01`, and so on through `20`.
- Both conditions use an image and empty text. The steering direction is B − A.

## Directory layout

```text
military_vehicle_condition/
├── README.md
├── manifest.json
└── images/
    ├── A_not_destroyed/
    │   ├── military_vehicle_A_01.jpg
    │   ├── military_vehicle_A_02.jpg
    │   └── … military_vehicle_A_20.jpg
    └── B_destroyed/
        ├── military_vehicle_B_01.jpg
        ├── military_vehicle_B_02.jpg
        └── … military_vehicle_B_20.jpg
```

**The image files shown above are expected filenames, not included assets.** The folders contain only `.gitkeep` files until you add your images.

## Add your images

1. Put each not-destroyed image in `images/A_not_destroyed/` and its destroyed counterpart in `images/B_destroyed/`.
2. Use the filenames above, or edit the corresponding `image` paths in `manifest.json`. PNG and other supported image formats can be used by updating the extensions in the manifest; do not simply rename a file's extension.
3. If you have fewer than 20 complete pairs, remove the unused entries from `pairs`. To add more, append entries with unique IDs and matching file paths. Every referenced image must exist before loading.
4. Restart the dashboard if it was already running when this directory was created. In **1 · Load inputs**, select `military_vehicle_condition/manifest.json` and click **Load repository manifest**, or upload the manifest JSON. Images remain on the server under its configured data root.
5. Continue with **2 · Common profile**. Reload the manifest after replacing images on disk.

Match comparable vehicle types, viewpoints, framing, lighting, and backgrounds where practical, so the contrast reflects vehicle condition. Use before/after views of the same vehicle when available; otherwise use comparable vehicles. Keep A/B labels consistent across all pairs.

The explicit `asset_root` makes repository selection and JSON upload resolve to the same image folders. Paths are relative to this use-case directory; no loader changes are needed.

## Pair format

```json
{
  "id": "military_vehicle_01",
  "A": {
    "text": "",
    "image": "images/A_not_destroyed/military_vehicle_A_01.jpg"
  },
  "B": {
    "text": "",
    "image": "images/B_destroyed/military_vehicle_B_01.jpg"
  }
}
```
