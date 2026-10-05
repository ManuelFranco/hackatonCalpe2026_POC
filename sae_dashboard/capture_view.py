"""Small, escaped preview of the regions actually used to build a profile."""

from html import escape

from research.code_regions import TOKEN_SCOPES

CAPTURE_CSS = """
.capture-preview {overflow-wrap:anywhere;}
.capture-preview pre {white-space:pre-wrap;overflow-wrap:anywhere;padding:14px;border:1px solid var(--border-color-primary);border-radius:8px;}
.capture-preview mark {background:rgba(99,102,241,.17);color:inherit;border-radius:2px;}
"""


def choices(session):
    return [(row["label"], str(i)) for i, row in enumerate(session.capture_details)]


def render(session, selected="0"):
    if not session.capture_details or selected is None:
        return ""
    try:
        row = session.capture_details[int(selected)]
    except (ValueError, IndexError, TypeError):
        raise ValueError("Select a captured input from the current profile.") from None
    text = row["text"]
    parts, cursor = [], 0
    for start, end in row.get("spans", []):
        parts.extend(
            (escape(text[cursor:start]), "<mark>", escape(text[start:end]), "</mark>")
        )
        cursor = end
    parts.append(escape(text[cursor:]))
    note = (
        "Highlighted source regions map to the selected tokens; boundary tokens can include adjacent whitespace."
        if row.get("spans")
        else "This scope selects chat tokens; no source-region highlighting is applied."
    )
    return (
        f'<div class="capture-preview"><p><strong>{escape(TOKEN_SCOPES[row["scope"]])}</strong> · '
        f"{row['selected_tokens']}/{row['total_tokens']} input tokens</p><pre>{''.join(parts)}</pre><small>{note}</small></div>"
    )
