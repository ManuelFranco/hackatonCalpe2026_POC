"""Code-region selection and exact text-token alignment. No inference or execution."""

import ast
from dataclasses import dataclass
import re


TOKEN_SCOPES = {
    "all": "All tokens",
    "last": "Last token",
    "non_image": "Non-image tokens",
    "code": "Code only",
    "sql_query": "SQL construction",
    "sql_call": "SQL execution calls",
    "sql_flow": "SQL construction + calls",
}
REGION_SCOPES = frozenset({"code", "sql_query", "sql_call", "sql_flow"})
SQL_METHODS = frozenset({"execute", "executemany", "executescript"})


@dataclass(frozen=True)
class Block:
    start: int
    text: str
    language: str


def code_blocks(text):
    """Keep exact character positions, including Unicode and CRLF inputs."""
    opening = re.compile(r"(?m)^[ \t]*(`{3,}|~{3,})([^\r\n]*)\r?\n")
    blocks, position = [], 0
    while match := opening.search(text, position):
        marker = match[1]
        closing = re.compile(
            rf"(?m)^[ \t]*{re.escape(marker[0])}{{{len(marker)},}}[ \t]*\r?(?:\n|$)"
        ).search(text, match.end())
        if closing is None:
            raise ValueError("Close every code fence before using code-region capture.")
        blocks.append(
            Block(
                match.end(),
                text[match.end() : closing.start()],
                match[2].strip().lower(),
            )
        )
        position = closing.end()
    if not blocks:
        # Unfenced inputs must actually parse as Python or look like C source.
        try:
            ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            if not re.search(r"(?m)^\s*#\s*include\b", text):
                raise ValueError(
                    "Code-region capture needs fenced code or unfenced source code."
                ) from None
        blocks = [Block(0, text, "")]
    return blocks


def merge_spans(spans):
    result = []
    for start, end in sorted(set(spans)):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _node_span(block, node):
    lines = block.text.splitlines(keepends=True)

    # AST columns are UTF-8 byte offsets; tokenizer offsets are characters.
    def position(line, column):
        prefix = lines[line - 1].encode("utf-8")[:column].decode("utf-8")
        return block.start + sum(len(s) for s in lines[: line - 1]) + len(prefix)

    return position(node.lineno, node.col_offset), position(
        node.end_lineno, node.end_col_offset
    )


def select_regions(text, scope):
    if scope not in REGION_SCOPES:
        raise ValueError("Choose a code-region scope.")
    blocks = code_blocks(text)
    if scope == "code":
        spans = merge_spans((b.start, b.start + len(b.text)) for b in blocks)
    else:
        calls, construction = [], []
        for block in blocks:
            if block.language not in {"", "py", "python", "python3"}:
                continue
            try:
                tree = ast.parse(block.text)
            except (SyntaxError, ValueError, RecursionError):
                raise ValueError(
                    "SQL-region capture requires valid Python code."
                ) from None
            parents = {
                child: parent
                for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)
            }
            for call in ast.walk(tree):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in SQL_METHODS
                ):
                    continue
                calls.append(_node_span(block, call))
                statement = call
                while statement in parents and not isinstance(statement, ast.stmt):
                    statement = parents[statement]
                owner = parents.get(statement)
                siblings = (
                    next(
                        (
                            v
                            for _, v in ast.iter_fields(owner)
                            if isinstance(v, list) and statement in v
                        ),
                        [],
                    )
                    if owner
                    else []
                )
                cutoff = siblings.index(statement) if siblings else 0

                def collect(expression, before, visited):
                    construction.append(_node_span(block, expression))
                    for name in {
                        n.id for n in ast.walk(expression) if isinstance(n, ast.Name)
                    }:
                        for index in range(before - 1, -1, -1):
                            assignment = siblings[index]
                            if (
                                isinstance(assignment, ast.Assign)
                                and len(assignment.targets) == 1
                            ):
                                target, value = assignment.targets[0], assignment.value
                            elif (
                                isinstance(assignment, ast.AnnAssign)
                                and assignment.value is not None
                            ):
                                target, value = assignment.target, assignment.value
                            else:
                                continue
                            if isinstance(target, ast.Name) and target.id == name:
                                key = (index, name)
                                if key not in visited:
                                    visited.add(key)
                                    construction.append(_node_span(block, assignment))
                                    collect(value, index, visited)
                                break

                query = (
                    call.args[0]
                    if call.args
                    else next(
                        (k.value for k in call.keywords if k.arg in {"sql", "query"}),
                        None,
                    )
                )
                if query is not None:
                    collect(query, cutoff, set())
        if not calls:
            raise ValueError(
                "No Python execute / executemany / executescript call found. Use Code only for other families."
            )
        spans = merge_spans(
            calls
            if scope == "sql_call"
            else construction
            if scope == "sql_query"
            else calls + construction
        )
    if not spans:
        raise ValueError("The selected code region is empty.")
    return spans


def aligned_token_mask(tokenizer, rendered, input_ids, prompt, spans):
    """Retokenize the complete chat and require identical IDs before using offsets."""
    source, source_start = prompt, 0
    if rendered.count(source) != 1:
        # Gemma's chat template trims message text. Keep spans in original-source
        # coordinates and account only for removed boundary whitespace.
        source = prompt.strip()
        source_start = len(prompt) - len(prompt.lstrip())
    if not source or rendered.count(source) != 1:
        raise ValueError(
            "The source text cannot be uniquely located in the chat template."
        )
    try:
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
    except (TypeError, ValueError, NotImplementedError):
        raise ValueError(
            "Code-region capture requires a tokenizer with character offsets."
        ) from None
    expected = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    if list(encoded["input_ids"]) != expected:
        raise ValueError(
            "Chat token IDs differ from the offset tokenizer. Region capture was stopped instead of guessing alignment."
        )
    offset = rendered.index(source)
    source_end = source_start + len(source)
    absolute = [
        (offset + left - source_start, offset + right - source_start)
        for start, end in spans
        if (left := max(start, source_start)) < (right := min(end, source_end))
    ]
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(expected):
        raise ValueError("Tokenizer offsets have the wrong length.")
    mask = [
        end > start and any(start < right and end > left for left, right in absolute)
        for start, end in offsets
    ]
    if not any(mask):
        raise ValueError("No tokens overlap the selected code region.")
    return mask
