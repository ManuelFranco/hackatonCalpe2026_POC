"""Bounded Python syntax / direct SQL-construction review. Never executes code."""

import ast
import re


def python_blocks(text):
    blocks, lines, start, language, fence = [], [], 1, None, None
    for lineno, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if match and fence is None:
            fence, language = match[1], match[2].strip().lower()
            lines, start = [], lineno + 1
        elif (
            match
            and match[1][0] == fence[0]
            and len(match[1]) >= len(fence)
            and not match[2].strip()
        ):
            if language in {"", "py", "python", "python3"}:
                blocks.append((start, "\n".join(lines), False))
            fence = None
        elif fence is not None:
            lines.append(line)
    if fence is not None and language in {"", "py", "python", "python3"}:
        blocks.append((start, "\n".join(lines), True))
    if language is None:
        blocks.append((1, text, False))
    return blocks


def review_python(text):
    blocks, findings = [], []
    for start, code, unclosed in python_blocks(text):
        block = {"start_line": start, "unclosed_fence": unclosed, "syntax": "Parsed"}
        try:
            if not code.strip():
                raise SyntaxError("Empty code block")
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError) as error:
            block.update(syntax="Not parsed", error=str(error))
            blocks.append(block)
            continue
        blocks.append(block)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"execute", "executemany", "executescript"}
            ):
                continue
            query = node.args[0] if node.args else None
            origin = None
            # Resolve only an immediately preceding assignment in the same block.
            # Branches, aliases and earlier assignments deliberately stay unresolved.
            if isinstance(query, ast.Name) and isinstance(node.func.value, ast.Name):
                statement = node
                while statement in parents and not isinstance(statement, ast.stmt):
                    statement = parents[statement]
                owner = parents.get(statement)
                for _, siblings in ast.iter_fields(owner) if owner else ():
                    if not isinstance(siblings, list) or statement not in siblings:
                        continue
                    index = siblings.index(statement)
                    previous = siblings[index - 1] if index else None
                    if (
                        isinstance(previous, ast.Assign)
                        and len(previous.targets) == 1
                        and isinstance(previous.targets[0], ast.Name)
                        and previous.targets[0].id == query.id
                    ):
                        origin = {"name": query.id, "line": start + previous.lineno - 1}
                        query = previous.value
                    break
            if isinstance(query, ast.Constant) and isinstance(query.value, str):
                kind = "Literal query"
            elif isinstance(query, (ast.JoinedStr, ast.BinOp)) or (
                isinstance(query, ast.Call)
                and isinstance(query.func, ast.Attribute)
                and query.func.attr == "format"
            ):
                kind = "Dynamic query · review"
            else:
                kind = "Unresolved query"
            expression = query if query is not None else node
            finding = {
                "line": start + node.lineno - 1,
                "kind": kind,
                "query_lines": [
                    start + expression.lineno - 1,
                    start + expression.end_lineno - 1,
                ],
                "call_lines": [start + node.lineno - 1, start + node.end_lineno - 1],
            }
            if origin:
                finding["assignment"] = origin
            findings.append(finding)
    return {
        "review_version": 3,
        "blocks": blocks,
        "findings": sorted(findings, key=lambda f: f["line"]),
        "syntax": "Not assessed"
        if not blocks
        else (
            "Parsed" if all(b["syntax"] == "Parsed" for b in blocks) else "Not parsed"
        ),
        "dynamic_queries": sum(f["kind"] == "Dynamic query · review" for f in findings),
        "unresolved_queries": sum(f["kind"] == "Unresolved query" for f in findings),
        "executed": False,
    }
