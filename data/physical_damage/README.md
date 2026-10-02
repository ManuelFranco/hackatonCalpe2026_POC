# Physical vehicle damage

Sixteen matched vehicle image pairs from the physical damage dataset. **A is
intact; B shows visible physical damage.** Conditions contain images only, matching
the other image datasets in the application. Original PNG bytes and pair IDs are
preserved.

| Repository manifest | Pairs | IDs | Intended use |
| --- | ---: | --- | --- |
| [manifest.json](manifest.json) | 8 | Par1–Par8 | Build the common profile and steering vectors |
| [manifest.validation.json](manifest.validation.json) | 3 | Par9–Par11 | Validation images |
| [manifest.test.json](manifest.test.json) | 5 | HoldoutPart1–HoldoutPart5 | Test images |

In **1 · Load inputs**, choose `physical_damage/manifest.json` from **Repository
manifest** and click **Load repository manifest**. The existing input preview shows
both images for each pair. Build the profile and vectors in sections 2 and 3.

The application computes **B − A**. Positive layer strengths steer toward the
damaged condition; negative strengths steer toward the intact condition. This
describes the vector direction, not a guarantee that the answer will change.

For comparisons in **4 · Base vs. steered**, upload individual validation or test
images from `images/A_intact/` or `images/B_damaged/` and use the same question and
generation settings. Keep the profile built from the training manifest. Loading
another manifest clears the current profile and vectors.

The `split` fields document the prior assignments. The generic application loader
does not enforce partitions: building a profile uses every pair in the loaded
manifest. Keep validation and test manifests out of profile building if those
images are to remain reserved for evaluation. They can be loaded separately to
browse the images.

`asset_root: "physical_damage"` makes uploaded copies of these manifests resolve
the same assets under `GEMMA_DATA_ROOT`. Filenames retain their original `train`
and `val` suffixes, but the explicit manifests determine the partitions. Par9–Par11
remain validation despite their filenames; the five `Part*-val-*` pairs remain
TEST. Pair roles come from the latest reviewed manifests, rather than filename
inference.
