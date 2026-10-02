# Military Vehicle Fire Condition - Image Pairs

One use case with 7 numbered A/B image pairs, following the paired CWE dataset convention.

- **A · Not in fire:** a vehicle that is not on fire.
- **B · In fire:** a vehicle that is on fire.
- Pair `01` matches A `01` with B `01`, and so on through `07`.
- Both conditions use an image and empty text. The steering direction is B − A.

## Directory layout

```text
military_vehicle_condition/
├── README.md
├── manifest.json
└── images/
    ├── A_not_in_fire/
    │   ├── military_vehicle_A_01.png
    │   ├── military_vehicle_A_02.png
    │   └── ... military_vehicle_A_07.png
    └── B_in_fire/
      ├── military_vehicle_B_01.png
      ├── military_vehicle_B_02.png
      └── ... military_vehicle_B_07.png
```

  ## Load the images

  1. Put each not-in-fire image in `images/A_not_in_fire/` and its in-fire counterpart in `images/B_in_fire/`.
2. Use the filenames above, or edit the corresponding `image` paths in `manifest.json`. PNG and other supported image formats can be used by updating the extensions in the manifest; do not simply rename a file's extension.
  3. To add more pairs, append entries with unique IDs and matching file paths. Every referenced image must exist before loading.
4. Restart the dashboard if it was already running when this directory was created. In **1 · Load inputs**, select `military_vehicle_condition/manifest.json` and click **Load repository manifest**, or upload the manifest JSON. Images remain on the server under its configured data root.
5. Continue with **2 · Common profile**. Reload the manifest after replacing images on disk.

Match comparable vehicle types, viewpoints, framing, lighting, and backgrounds where practical, so the contrast reflects the presence of fire. Keep A/B labels consistent across all pairs.

The explicit `asset_root` makes repository selection and JSON upload resolve to the same image folders. Paths are relative to this use-case directory; no loader changes are needed.

## Pair format

```json
{
  "id": "military_vehicle_01",
  "A": {
    "text": "",
    "image": "images/A_not_in_fire/military_vehicle_A_01.png"
  },
  "B": {
    "text": "",
    "image": "images/B_in_fire/military_vehicle_B_01.png"
  }
}
```
