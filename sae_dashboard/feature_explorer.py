"""Read-only profile diagnostics and expandable Neuronpedia feature tables.

Overlap is diagnostic: this module never masks or modifies research vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import math
import torch
from research.vector_builders import LayerProfile, VectorResult

PRESENCE_EPSILON = 1e-6
PREVIEW_LIMIT = 20
NEURONPEDIA_MODEL = "gemma-3-4b-it"
NEURONPEDIA_SOURCES = {i: f"{i}-gemmascope-2-res-16k" for i in (9, 17, 22, 29)}
EMBED_QUERY = "embed=true&embedexplanation=true&embedplots=true&embedsteer=true&embedactivations=false&embedlink=true&embedtest=true"

# Load only the feature the user expands, not every iframe in all four layers.
NEURONPEDIA_JS = """
element.addEventListener('toggle', (event) => {
    const row = event.target;
    if (!row.matches?.('details.feature-row') || !row.open) return;
    const frame = row.querySelector('iframe[data-src]');
    if (frame && !frame.hasAttribute('src')) frame.src = frame.dataset.src;
}, true);
"""

FEATURE_CSS = """
.profile-explorer {min-width:0;}
.profile-explorer .layer-stat-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin:8px 0 16px;}
.profile-explorer .layer-stat {border:1px solid rgba(127,127,127,.2);background:rgba(127,127,127,.045);border-radius:12px;padding:12px;}
.profile-explorer .stat-value {display:block;font-size:1.5rem;font-weight:750;font-variant-numeric:tabular-nums;}
.profile-explorer .stat-label {display:block;font-size:.82rem;line-height:1.35;opacity:.85;}
.profile-explorer .layer-info {font-size:.9rem;margin:10px 0 16px;overflow-wrap:anywhere;}
.profile-explorer .layer-table-caption {margin:16px 0 8px;}
.profile-explorer .layer-table-caption h4 {font-size:1rem;font-weight:700;margin:0;}
.profile-explorer .layer-table-caption small {opacity:.8;}
.profile-explorer .feature-table {max-width:100%;overflow-x:auto;border:1px solid rgba(127,127,127,.2);border-radius:12px;}
.profile-explorer .feature-grid {display:grid;grid-template-columns:80px 85px 90px 90px 100px 105px 110px 115px minmax(175px,1fr) 100px;gap:10px;align-items:center;min-width:1150px;padding:10px 12px;font-size:.85rem;}
.profile-explorer .feature-grid-head {font-size:.73rem;text-transform:uppercase;letter-spacing:.03em;font-weight:750;background:rgba(127,127,127,.08);}
.profile-explorer .feature-row {border-top:1px solid rgba(127,127,127,.15);}
.profile-explorer .feature-row > summary {cursor:pointer;list-style:none;}
.profile-explorer .feature-row > summary::-webkit-details-marker {display:none;}
.profile-explorer .feature-row:nth-child(even) > summary {background:rgba(127,127,127,.025);}
.profile-explorer .feature-row > summary:hover,.profile-explorer .feature-row[open] > summary {background:rgba(99,102,241,.08);}
.profile-explorer .feature-id {display:inline-block;padding:4px 8px;border-radius:8px;font-weight:750;font-variant-numeric:tabular-nums;}
.profile-explorer .feature-id.pos {background:rgba(16,185,129,.14);}
.profile-explorer .feature-id.neg {background:rgba(244,63,94,.12);}
.profile-explorer .signed-value {font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap;}
.profile-explorer .pos {color:#087c4b;}
.profile-explorer .neg {color:#c23546;}
.profile-explorer .zero {opacity:.7;}
.profile-explorer .feature-presence,.profile-explorer .feature-chip {display:inline-block;padding:4px 7px;border-radius:8px;background:rgba(127,127,127,.1);font-size:.8rem;}
.profile-explorer .feature-presence.full {background:rgba(16,185,129,.12);color:#087c4b;}
.profile-explorer .feature-chip.ambiguous {background:rgba(245,158,11,.15);color:#805200;}
.profile-explorer .feature-pairs {font-variant-numeric:tabular-nums;font-size:.8rem;overflow-wrap:anywhere;}
.profile-explorer .np-trigger {font-size:.75rem;font-weight:700;padding:5px;border:1px solid rgba(99,102,241,.3);border-radius:7px;text-align:center;}
.profile-explorer .np-trigger:before {content:'+ ';}
.profile-explorer .feature-row[open] .np-trigger:before {content:'− ';}
.profile-explorer .np-panel {padding:16px;border-top:1px solid rgba(127,127,127,.15);}
.profile-explorer .np-iframe {display:block;width:100%;max-width:900px;height:360px;border:1px solid rgba(127,127,127,.2);border-radius:10px;background:white;}
.profile-explorer .np-external {display:inline-block;margin-top:8px;text-decoration:underline;}
.profile-explorer .feature-empty {padding:18px;opacity:.8;}
.profile-explorer .feature-help {margin:12px 0;}
.profile-explorer .feature-help > summary {cursor:pointer;font-weight:600;}
.dark .profile-explorer .pos,.dark .profile-explorer .feature-presence.full {color:#5ce0aa;}
.dark .profile-explorer .neg {color:#ff8590;}
.dark .profile-explorer .feature-chip.ambiguous {color:#ffc875;}
@media (prefers-color-scheme:dark) {
 .profile-explorer .pos,.profile-explorer .feature-presence.full {color:#5ce0aa;}
 .profile-explorer .neg {color:#ff8590;}
 .profile-explorer .feature-chip.ambiguous {color:#ffc875;}
}
"""


@dataclass(frozen=True)
class FeatureAnalysis:
    a: torch.Tensor
    b: torch.Tensor
    a_mean: torch.Tensor
    b_mean: torch.Tensor
    deltas: torch.Tensor
    mean: torch.Tensor
    final: torch.Tensor | None
    masks: dict[str, torch.Tensor]
    presence_count: torch.Tensor
    stats: dict[str, int | None]


def analyze_profile(
    profile: LayerProfile,
    vector: VectorResult | None = None,
    epsilon: float = PRESENCE_EPSILON,
) -> FeatureAnalysis:
    a, b = profile.a.detach().float().cpu(), profile.b.detach().float().cpu()
    if a.ndim != 2 or a.shape != b.shape or not a.shape[0] or not a.shape[1]:
        raise ValueError(
            "Profile features must have matching nonempty [pairs, features] shapes."
        )
    if (
        not math.isfinite(epsilon)
        or epsilon < 0
        or not bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    ):
        raise ValueError("Profile scores and presence threshold must be finite.")
    delta = b - a
    presence = delta.abs() > epsilon
    counts = presence.sum(0)
    any_delta, common = counts > 0, presence.all(0)
    mean = delta.mean(0)
    # Threshold is for diagnostics only. The baseline retains all feature values.
    active_a, active_b = a.abs().gt(epsilon).any(0), b.abs().gt(epsilon).any(0)
    opposing = delta.gt(epsilon).any(0) & delta.lt(-epsilon).any(0)
    canceled = any_delta & mean.abs().le(epsilon)
    final = None if vector is None else vector.feature_delta.detach().float().cpu()
    if final is not None and (
        final.shape != mean.shape or not bool(torch.isfinite(final).all())
    ):
        raise ValueError("Vector features must match the profile and be finite.")
    # Removed means explicitly zeroed by the actual method, not absent in overlap.
    removed = torch.zeros_like(common) if final is None else mean.ne(0) & final.eq(0)
    masks = {
        "common": common,
        "outside": any_delta & ~common,
        "canceled": canceled,
        "removed": removed,
        "opposing": opposing,
        "active": active_a | active_b | any_delta,
    }
    stats = {
        "pairs": a.shape[0],
        "total": a.shape[1],
        "active_a": int(active_a.sum()),
        "active_b": int(active_b.sum()),
        "active_either": int((active_a | active_b).sum()),
        "active_both": int((active_a & active_b).sum()),
        "any_delta": int(any_delta.sum()),
        "common": int(common.sum()),
        "outside": int(masks["outside"].sum()),
        "never_active": int((~(active_a | active_b)).sum()),
        "opposing_common": int((common & opposing).sum()),
        "canceled": int(canceled.sum()),
        "mean_nonzero": int(mean.ne(0).sum()),
        "removed": None if final is None else int(removed.sum()),
        "final_nonzero": None if final is None else int(final.ne(0).sum()),
    }
    return FeatureAnalysis(
        a, b, a.mean(0), b.mean(0), delta, mean, final, masks, counts, stats
    )


def neuronpedia_url(layer: int, feature: int, dictionary_size: int = 16384) -> str:
    if layer not in NEURONPEDIA_SOURCES or not isinstance(feature, int) or feature < 0:
        raise ValueError("Unconfigured Neuronpedia layer or invalid feature ID.")
    width = {16384: "16k", 262144: "262k"}.get(dictionary_size)
    if width is None:
        raise ValueError("No configured Neuronpedia source for this dictionary size.")
    return f"https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/{layer}-gemmascope-2-res-{width}/{feature}?{EMBED_QUERY}"


def signed(value: float) -> str:
    color = "pos" if value > 0 else "neg" if value < 0 else "zero"
    return f'<span class="signed-value {color}">{value:+.6g}</span>'


def empty_profile(layer: int) -> str:
    return f'<div class="profile-explorer"><div class="feature-empty">Layer {layer}: build a common profile to inspect features.</div></div>'


def _table(analysis, layer, title, mask, pair_labels, neuronpedia, limit=PREVIEW_LIMIT):
    ids = mask.nonzero(as_tuple=True)[0]
    # Rank with tensors; only the small preview is converted into Python/HTML.
    peak = analysis.deltas[:, ids].abs().amax(0)
    order = torch.argsort(peak, descending=True, stable=True)[:limit]
    ids = ids[order].tolist()
    parts = [
        f'<div class="layer-table-caption"><h4>{html.escape(title)}</h4>'
        f"<small>Showing {len(ids)} of {int(mask.sum()):,} · ranked by peak |B − A|</small></div>"
    ]
    if not ids:
        return (
            "".join(parts)
            + '<div class="feature-empty">No features in this category.</div>'
        )
    columns = [
        "Feature",
        "Presence",
        "Mean A",
        "Mean B",
        "Mean B − A",
        "Vector Δ",
        "Signs",
        "Status",
        "Pair Δ values",
        "Neuronpedia",
    ]
    parts.append(
        '<div class="feature-table"><div class="feature-grid feature-grid-head">'
        + "".join(f"<span>{c}</span>" for c in columns)
        + "</div>"
    )
    for fid in ids:
        mean = float(analysis.mean[fid])
        count, total = int(analysis.presence_count[fid]), analysis.deltas.shape[0]
        color = "pos" if mean > 0 else "neg" if mean < 0 else "zero"
        opposite, canceled, removed = (
            bool(analysis.masks[key][fid])
            for key in ("opposing", "canceled", "removed")
        )
        signs = (
            "Opposite signs" if opposite else "Same sign" if count else "No contrast"
        )
        status = (
            "Removed"
            if removed
            else "Canceled mean"
            if canceled
            else "Preview"
            if analysis.final is None
            else "Retained"
            if analysis.final[fid] != 0
            else "Zero"
        )
        pair_values = analysis.deltas[:, fid].tolist()
        # A compact preview plus an expanded table retains every pair, including long manifests.
        brief = ", ".join(signed(v) for v in pair_values[:4]) + (
            " …" if total > 4 else ""
        )
        contribution = (
            "<span>Not built</span>"
            if analysis.final is None
            else signed(float(analysis.final[fid]))
        )
        parts.extend(
            [
                '<details class="feature-row"><summary class="feature-grid">',
                f'<span><span class="feature-id {color}">#{fid}</span></span>',
                f'<span><span class="feature-presence {"full" if count == total else ""}">{count}/{total}</span></span>',
                f"<span>{float(analysis.a_mean[fid]):.6g}</span>",
                f"<span>{float(analysis.b_mean[fid]):.6g}</span>",
                f"<span>{signed(mean)}</span>",
                f"<span>{contribution}</span>",
                f'<span class="feature-chip {"ambiguous" if opposite else ""}">{signs}</span>',
                f'<span class="feature-chip {"ambiguous" if canceled or removed else ""}">{status}</span>',
                f'<span class="feature-pairs">{brief}</span>',
                '<span class="np-trigger">EXPAND</span></summary>',
                '<div class="np-panel">',
            ]
        )
        if neuronpedia and analysis.a.shape[1] in (16384, 262144):
            url = html.escape(
                neuronpedia_url(layer, fid, analysis.a.shape[1]), quote=True
            )
            parts.extend(
                [
                    f'<iframe class="np-iframe" data-src="{url}" title="Neuronpedia · layer {layer}, feature {fid}" loading="lazy" referrerpolicy="strict-origin-when-cross-origin" allow="clipboard-write"></iframe>',
                    f'<a class="np-external" href="{url}" target="_blank" rel="noopener noreferrer">Open in Neuronpedia ↗</a>',
                ]
            )
        else:
            parts.append("<p>Neuronpedia is not configured for this model / SAE.</p>")
        parts.append(
            '<details class="feature-help"><summary>All pair values</summary><table><thead><tr><th>Pair</th><th>A</th><th>B</th><th>B − A</th></tr></thead><tbody>'
        )
        for index, value in enumerate(pair_values):
            label = (
                pair_labels[index] if index < len(pair_labels) else f"Pair {index + 1}"
            )
            parts.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td>{float(analysis.a[index, fid]):.6g}</td>"
                f"<td>{float(analysis.b[index, fid]):.6g}</td>"
                f"<td>{signed(value)}</td></tr>"
            )
        parts.append("</tbody></table></details></div></details>")
    parts.append("</div>")
    return "".join(parts)


def render_profile(
    layer: int,
    profile: LayerProfile,
    vector: VectorResult | None = None,
    pair_labels: list[str] | None = None,
    neuronpedia: bool = True,
) -> str:
    analysis = analyze_profile(profile, vector)
    stats = analysis.stats
    cards = [
        ("Active in A", "active_a"),
        ("Active in B", "active_b"),
        ("Active in A or B", "active_either"),
        ("Active in both A and B", "active_both"),
        ("Changed in any pair", "any_delta"),
        ("Overlapping in every pair", "common"),
        ("Outside overlap", "outside"),
        ("Opposite signs in overlap", "opposing_common"),
        ("Canceled after averaging", "canceled"),
        ("Removed by vector method", "removed"),
        ("Nonzero vector features", "final_nonzero"),
        ("Never active", "never_active"),
    ]
    body = ['<div class="profile-explorer"><div class="layer-stat-grid">']
    for label, key in cards:
        number = "—" if stats[key] is None else f"{stats[key]:,}"
        body.append(
            f'<div class="layer-stat"><span class="stat-value">{number}</span><span class="stat-label">{label}</span></div>'
        )
    body.append(
        '</div><div class="layer-info">'
        f"Layer {layer} · {stats['pairs']} pairs · {stats['total']:,} total features. "
        '<span class="signed-value pos">Green: positive B − A</span> · '
        '<span class="signed-value neg">Red: negative B − A</span>. Expand a feature for Neuronpedia.</div>'
    )
    body.append(
        '<details class="feature-help"><summary>How to read these counts</summary>'
        f"<p>Activity and contrast presence use |value| &gt; {PRESENCE_EPSILON:g}. Activity in A/B means active in at least one pair. "
        "Overlap means a B − A contrast in every pair, regardless of sign. Outside overlap is a diagnostic count, not a removal rule. "
        "Canceled means have contrast in a pair but |mean B − A| ≤ ε. The vector method decides which features are removed; "
        "the baseline retains all mean differences. Vector Δ is the feature-space value before decoding and scaling, not the layer strength. "
        "Vector counters show — until vectors are built.</p></details>"
    )
    labels = pair_labels or []
    body.append(
        _table(
            analysis,
            layer,
            "Common features · overlapping across pairs",
            analysis.masks["common"],
            labels,
            neuronpedia,
        )
    )
    body.append(
        _table(
            analysis,
            layer,
            "Non-common features · outside overlap",
            analysis.masks["outside"],
            labels,
            neuronpedia,
        )
    )
    if bool(analysis.masks["removed"].any()):
        body.append(
            _table(
                analysis,
                layer,
                "Features removed by vector method",
                analysis.masks["removed"],
                labels,
                neuronpedia,
            )
        )
    body.append("</div>")
    return "".join(body)
