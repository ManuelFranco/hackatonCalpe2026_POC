"""Build and audit the repository's synthetic SQLite A/B fixtures offline.

No model, third-party packages, network access, or persistent database is needed.
Build preserves pair IDs and splits, with one changed line per A/B pair.
Audit checks actual SQLite behavior rather
than treating the presence of formatting or placeholders as the reference label.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "sql_injection"
REVISION = "code-flow-v4-single-line-20"
PAIR_COUNTS = {"train": 20, "validation": 20, "test": 20}
CONTEXT = "SQLite. db is a database connection; all other arguments are strings from an HTTP request."
TASK = "In three sentences, explain how the supplied values reach the database call and affect the executed query. Suggest a change only if needed."
FORMATS = ("f_string", "concatenation", "percent", "str_format")
PREDICATE = "missing' OR '1' LIKE '1"
MANIFESTS = {
    "train": "manifest.json",
    "validation": "manifest.validation.json",
    "test": "manifest.test.json",
}
LEGACY_STEMS = {
    "train": (
        "lookup_catalog", "lookup_tickets", "lookup_devices", "lookup_products",
        "lookup_books", "lookup_packages", "lookup_builds", "lookup_hosts",
        "lookup_services", "lookup_projects", "lookup_sessions", "lookup_artifacts",
    ),
    "validation": ("cursor_lookup", "named_binding", "normalized_input", "prefix_search"),
    "test": (
        "allowlisted_identifier", "display_interpolation", "two_values",
        "named_reuse", "substring_search", "query_reassignment",
    ),
}
TRAIN_FAMILIES = (
    "equality", "inline_call", "cursor_result", "named_dictionary", "numbered_reuse",
    "preview_direct", "preview_starred", "like_value", "normalized_value",
    "partial_binding", "named_repetition", "selected_alias", "late_assignment",
    "mapped_identifier", "list_membership", "cte_predicate", "join_predicate",
    "nested_predicate", "insert_value", "update_predicate", "delete_predicate",
    "batch_update",
)
TRAIN_SELECTION = {
    "named_dictionary": 3,
    "like_value": 1,
    "partial_binding": 2,
    "mapped_identifier": 0,
    "list_membership": 3,
    "insert_value": 2,
    "update_predicate": 1,
    "delete_predicate": 0,
}
VALIDATION_FAMILIES = (
    "mapping_packet", "alias_chain", "aggregate_having", "case_predicate",
    "optional_filter", "insert_partial_binding", "update_set_value", "cursor_loop",
)
TEST_FAMILIES = (
    "indexed_packet", "history_selection", "union_partial_binding",
    "named_batch_preview", "like_escape", "join_list_partial_binding", "cte_two_paths",
)
DOMAINS = (
    "parts", "invoices", "memberships", "batches", "leases", "appointments",
    "messages", "entries", "tasks", "schedules", "labels", "revisions",
    "parcels", "directories", "contracts", "audits", "inventory", "queues",
    "submissions", "dispatches", "allocations", "samples", "events_log",
    "deliveries", "notifications", "registrations", "reservations", "attachments",
    "readings", "assignments", "receipts", "transactions_log", "resources",
    "checks", "summaries", "collections", "observations",
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def code(text):
    blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.S)
    if len(blocks) != 1:
        raise ValueError("Each fixture needs exactly one Python block.")
    ast.parse(blocks[0])
    return blocks[0]


def pair_contrast(a, b):
    """Require equal line counts and byte-identical context around one line."""
    lines_a, lines_b = a.splitlines(keepends=True), b.splitlines(keepends=True)
    if len(lines_a) != len(lines_b):
        raise ValueError("A/B inputs must differ on exactly one line, without inserted or removed lines.")
    changed = [i for i, (left, right) in enumerate(zip(lines_a, lines_b)) if left != right]
    if len(changed) != 1:
        raise ValueError("A/B inputs must differ on exactly one line.")
    index = changed[0]
    fragment = ast.parse(lines_a[index].strip())
    calls = [n for n in ast.walk(fragment) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"execute", "executemany"}]
    kind = "sql_execution" if calls else "sql_construction" if any(isinstance(n, (ast.JoinedStr, ast.Constant)) and isinstance(getattr(n, "value", None), str) and re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b", n.value) for n in ast.walk(fragment)) else "argument_selection"
    prefix = "".join(lines_a[:index])
    suffix = "".join(lines_a[index + 1:])
    if prefix != "".join(lines_b[:index]) or suffix != "".join(lines_b[index + 1:]):
        raise ValueError("A/B context differs outside the contrast line.")
    return {
        "line": index + 1,
        "kind": kind,
        "A": lines_a[index].rstrip("\r\n"),
        "B": lines_b[index].rstrip("\r\n"),
        "shared_prefix_sha256": digest(prefix.encode()),
        "shared_suffix_sha256": digest(suffix.encode()),
    }


def single_line_sources(sources):
    """Share setup and output handling; vary only the database operation.

    Existing one-line pairs are retained. For multi-line pairs, the A setup is
    shared, the B query/parameters are constructed alongside it, and one call
    selects the executed arguments. No unrelated statements are joined with ';'.
    """
    left, right = (sources[side].splitlines() for side in ("A", "B"))
    if len(left) == len(right) and sum(a != b for a, b in zip(left, right)) == 1:
        return sources
    trees = {side: ast.parse(sources[side]) for side in ("A", "B")}

    def sql_call(node):
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"execute", "executemany"}

    def sink(tree):
        calls = [n for n in ast.walk(tree) if sql_call(n)]
        if len(calls) != 1:
            raise ValueError("One-line adaptation requires one database call per source.")
        call = calls[0]
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        statement = call
        while not isinstance(statement, ast.stmt):
            statement = parents[statement]
        owner = parents[statement]
        block = next(value for _, value in ast.iter_fields(owner) if isinstance(value, list) and statement in value)
        return call, statement, block, block.index(statement)

    def resolve(expression, statements, before=None):
        cutoff = len(statements) if before is None else before

        class ExpandNames(ast.NodeTransformer):
            def visit_Name(self, node):
                if not isinstance(node.ctx, ast.Load):
                    return node
                for index in range(cutoff - 1, -1, -1):
                    assignment = statements[index]
                    if isinstance(assignment, ast.Assign) and len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name) and assignment.targets[0].id == node.id:
                        return resolve(assignment.value, statements, index)
                return node

        return ExpandNames().visit(copy.deepcopy(expression))

    call_a, statement_a, block_a, index_a = sink(trees["A"])
    call_b, _, block_b, index_b = sink(trees["B"])
    if not call_b.args or isinstance(call_b.args[0], ast.Starred):
        raise ValueError("Unsupported multi-line database argument shape.")
    identifiers = {n.id for tree in trees.values() for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def available_name(stem):
        name, number = stem, 2
        while name in identifiers:
            name, number = f"{stem}_{number}", number + 1
        identifiers.add(name)
        return name

    query_name = available_name("candidate")
    values_name = available_name("candidate_values")

    def assignment(name, expression):
        return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=expression)

    additions = []
    def query_branch(node, call):
        if not isinstance(node, ast.If) or not isinstance(call.args[0], ast.Name):
            return False
        return any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == call.args[0].id for t in n.targets) for n in ast.walk(node))

    branches_a = [n for n in block_a[:index_a] if query_branch(n, call_a)]
    branches_b = [n for n in block_b[:index_b] if query_branch(n, call_b)]
    changed_call = copy.deepcopy(call_b)
    if branches_a or branches_b:
        if len(branches_a) != 1 or len(branches_b) != 1 or len(call_b.args) != 2:
            raise ValueError("Unsupported branching database fixture.")
        # Construct both candidates in whichever branch the request takes.
        # The database operation stays outside the branch on one contrast line.
        for arm in ("body", "orelse"):
            original_b = getattr(branches_b[0], arm)
            shared_a = getattr(branches_a[0], arm)
            shared_a.append(assignment(query_name, resolve(call_b.args[0], original_b)))
            shared_a.append(assignment(values_name, resolve(call_b.args[1], original_b)))
        changed_call.args = [ast.Name(id=query_name, ctx=ast.Load()), ast.Name(id=values_name, ctx=ast.Load())]
    else:
        additions.append(assignment(query_name, resolve(call_b.args[0], block_b, index_b)))
        changed_call.args[0] = ast.Name(id=query_name, ctx=ast.Load())
        if len(call_b.args) > 1:
            changed_call.args[1] = resolve(call_b.args[1], block_b, index_b)

    def replace_call(statement, replacement):
        class ReplaceCall(ast.NodeTransformer):
            def visit_Call(self, node):
                if sql_call(node):
                    return copy.deepcopy(replacement)
                return self.generic_visit(node)

        return ReplaceCall().visit(copy.deepcopy(statement))

    if isinstance(statement_a, ast.Return):
        result_name = available_name("result")
        contrast_a = assignment(result_name, copy.deepcopy(call_a))
        contrast_b = assignment(result_name, changed_call)
        tail = [replace_call(statement_a, ast.Name(id=result_name, ctx=ast.Load()))]
    else:
        contrast_a = copy.deepcopy(statement_a)
        contrast_b = replace_call(statement_a, changed_call)
        tail = []
    prefix = block_a[:index_a] + additions
    suffix = tail + block_a[index_a + 1:]
    block_a[:] = prefix + [contrast_a] + suffix
    normalized_a = ast.unparse(ast.fix_missing_locations(trees["A"])) + "\n"
    block_a[len(prefix)] = contrast_b
    normalized_b = ast.unparse(ast.fix_missing_locations(trees["A"])) + "\n"
    pair_contrast(normalized_a, normalized_b)
    return {"A": normalized_a, "B": normalized_b}


def expression(template, formatter):
    """Render the same SQL interpolation with four Python construction idioms."""
    names = re.findall(r"\{([^{}]+)\}", template)
    if formatter == "f_string":
        return "f" + repr(template)
    if formatter == "str_format":
        literal = re.sub(r"\{[^{}]+\}", "{}", template)
        return repr(literal) + ".format(" + ", ".join(names) + ")"
    if formatter == "percent":
        literal = re.sub(r"\{[^{}]+\}", "%s", template.replace("%", "%%"))
        return repr(literal) + " % (" + ", ".join(names) + ",)"
    pieces = re.split(r"(\{[^{}]+\})", template)
    return " + ".join(p[1:-1] if p.startswith("{") else repr(p) for p in pieces if p)


def source(body, arguments, variant):
    name = ("lookup", "read_rows", "load_rows", "find_rows")[variant % 4]
    return "def " + name + "(db, " + ", ".join(arguments) + "):\n" + textwrap.indent(body.strip() + "\n", "    ")


def make_spec(family, table, column, other, note, formatter, variant):
    """Explicit templates; metadata and behavioral witnesses never enter prompts."""
    bound = f"SELECT id FROM {table} WHERE {column} = ?"
    literal = f"SELECT id FROM {table} WHERE {column} = '{{value}}'"
    rendered = expression(literal, formatter)
    args, normal, witness = ["value"], [["alpha"], ["missing"]], [PREDICATE]
    matched, dynamic = False, False
    mechanism = "A binds the supplied value; B places it inside the executed SQL text."

    if family == "equality":
        a = f"query = {bound!r}\nreturn db.execute(query, (value,)).fetchall()"
        b = f"query = {rendered}\nreturn db.execute(query).fetchall()"
    elif family == "inline_call":
        a = f"return db.execute({bound!r}, (value,)).fetchall()"
        b = f"return db.execute({rendered}).fetchall()"
    elif family == "cursor_result":
        shared = "cursor = db.cursor()\n"
        a = shared + f"query = {bound!r}\nreturn cursor.execute(query, (value,)).fetchall()"
        b = shared + f"query = {rendered}\nreturn cursor.execute(query).fetchall()"
    elif family == "named_dictionary":
        a = f"query = {bound.replace('?', ':item')!r}\nparameters = {{'item': value}}\nreturn db.execute(query, parameters).fetchall()"
        b = f"query = {rendered}\nparameters = {{'item': value}}\nreturn db.execute(query, parameters).fetchall()"
        matched = True
        mechanism = "The same dictionary is present on both sides; B's SQL has no bound item placeholder and the extra dictionary entry is ignored by sqlite3."
    elif family == "numbered_reuse":
        q = f"SELECT id FROM {table} WHERE {column} = ?1 OR {other} = ?1"
        p = expression(f"SELECT id FROM {table} WHERE {column} = '{{value}}' OR {other} = '{{value}}'", formatter)
        a = f"query = {q!r}\nreturn db.execute(query, (value,)).fetchall()"
        b = f"query = {p}\nreturn db.execute(query).fetchall()"
    elif family in {"preview_direct", "preview_starred", "selected_alias", "late_assignment", "alias_chain", "mapping_packet", "indexed_packet", "history_selection"}:
        matched = True
        shared = f"statement = {bound!r}\npreview = {rendered}\nparameters = (value,)\n"
        if family == "preview_direct":
            a = shared + "rows = db.execute(statement, parameters).fetchall()\nreturn preview, rows"
            b = shared + "rows = db.execute(preview, ()).fetchall()\nreturn preview, rows"
        elif family == "preview_starred":
            a = shared + "request = (statement, parameters)\nreturn preview, db.execute(*request).fetchall()"
            b = shared + "request = (preview, ())\nreturn preview, db.execute(*request).fetchall()"
        elif family == "selected_alias":
            a = shared + "selected = statement\narguments = parameters\nreturn db.execute(selected, arguments).fetchall()"
            b = shared + "selected = preview\narguments = ()\nreturn db.execute(selected, arguments).fetchall()"
        elif family == "late_assignment":
            shared += "query = preview\n"
            a = shared + "query = statement\narguments = parameters\nreturn db.execute(query, arguments).fetchall()"
            b = shared + "query = preview\narguments = ()\nreturn db.execute(query, arguments).fetchall()"
        elif family == "alias_chain":
            shared += "first = statement\nsecond = preview\n"
            a = shared + "selected = first\nquery = selected\nreturn db.execute(query, parameters).fetchall()"
            b = shared + "selected = second\nquery = selected\nreturn db.execute(query, ()).fetchall()"
        elif family == "mapping_packet":
            a = shared + "packet = {'query': statement, 'values': parameters}\nreturn db.execute(packet['query'], packet['values']).fetchall()"
            b = shared + "packet = {'query': preview, 'values': ()}\nreturn db.execute(packet['query'], packet['values']).fetchall()"
        elif family == "indexed_packet":
            shared += "requests = ((statement, parameters), (preview, ()))\n"
            a = shared + "return db.execute(*requests[0]).fetchall()"
            b = shared + "return db.execute(*requests[1]).fetchall()"
        else:
            shared += "history = [(statement, parameters), (preview, ())]\n"
            a = shared + "selected = history[-2]\nquery, arguments = selected\nreturn db.execute(query, arguments).fetchall()"
            b = shared + "selected = history[-1]\nquery, arguments = selected\nreturn db.execute(query, arguments).fetchall()"
        mechanism = "Both sides construct identical bound and interpolated candidates; the selected execution argument determines the condition."
    elif family in {"like_value", "normalized_value", "like_escape"}:
        if family == "like_value":
            shared = "pattern = '%' + value + '%'\n"
            q = f"SELECT id FROM {table} WHERE {column} LIKE ?"
            p = expression(f"SELECT id FROM {table} WHERE {column} LIKE '{{pattern}}'", formatter)
            mechanism = "A binds the complete LIKE pattern; B interpolates it. LIKE wildcard matching is separate from SQL syntax injection."
        elif family == "like_escape":
            slash = chr(92)
            shared = f"escaped = value.replace({slash!r}, {slash * 2!r}).replace('%', {slash + '%'!r}).replace('_', {slash + '_'!r})\npattern = '%' + escaped + '%'\n"
            q = f"SELECT id FROM {table} WHERE {column} LIKE ? ESCAPE '\\'"
            p = expression(f"SELECT id FROM {table} WHERE {column} LIKE '{{pattern}}' ESCAPE '\\'", formatter)
            mechanism = "Escaping LIKE wildcards does not quote SQL string literals; A binds the pattern, while B still exposes apostrophes to the SQL parser."
        else:
            shared = "item = value.strip().casefold()\n"
            q = bound
            p = expression(literal.replace("{value}", "{item}"), formatter)
            normal = [[" ALPHA "], [" MISSING "]]
            mechanism = "Stripping whitespace and case folding do not neutralize SQL syntax; only A binds the transformed value."
        shared += f"statement = {q!r}\npreview = {p}\n"
        parameter = "item" if family == "normalized_value" else "pattern"
        a = shared + f"return db.execute(statement, ({parameter},)).fetchall()"
        b = shared + "return db.execute(preview, ()).fetchall()"
        matched = True
    elif family == "partial_binding":
        args, normal, witness = ["value", "other"], [["alpha", "open"], ["missing", "open"]], ["alpha", PREDICATE]
        q = f"SELECT id FROM {table} WHERE {column} = ? AND {other} = ?"
        p = expression(f"SELECT id FROM {table} WHERE {column} = ? AND {other} = '{{other}}'", formatter)
        a = f"query = {q!r}\nreturn db.execute(query, (value, other)).fetchall()"
        b = f"query = {p}\nreturn db.execute(query, (value,)).fetchall()"
        mechanism = "Binding the first predicate does not protect the second interpolated value."
    elif family == "named_repetition":
        q = f"SELECT id FROM {table} WHERE {column} = :item OR {column} = :item"
        p = expression(f"SELECT id FROM {table} WHERE {column} = '{{value}}' OR {column} = '{{value}}'", formatter)
        shared = f"statement = {q!r}\npreview = {p}\nparameters = {{'item': value}}\n"
        a = shared + "return db.execute(statement, parameters).fetchall()"
        b = shared + "return db.execute(preview, parameters).fetchall()"
        matched = True
    elif family == "mapped_identifier":
        args, normal, witness = ["table_name", "value"], [["current", "alpha"], ["archive", "missing"]], ["current", PREDICATE]
        shared = f"tables = {{'current': {table!r}, 'archive': {table + '_copy'!r}}}\nselected = tables[table_name]\n"
        a = shared + f"query = f\"SELECT id FROM {{selected}} WHERE {column} = ?\"\nreturn db.execute(query, (value,)).fetchall()"
        # The only dynamic identifier comes from the same fixed mapping on B.
        p = expression(f"SELECT id FROM {{selected}} WHERE {column} = '{{value}}'", formatter)
        b = shared + f"query = {p}\nreturn db.execute(query).fetchall()"
        mechanism = "A fixed identifier mapping controls the table name; it does not protect B's interpolated predicate value."
        dynamic = True
    elif family == "list_membership":
        normal, witness = [["alpha,beta"], ["missing,unknown"]], ["missing') OR ('1' LIKE '1"]
        shared = "items = value.split(',')\nslots = ', '.join('?' for item in items)\nliterals = ', '.join(\"'\" + item + \"'\" for item in items)\n"
        q = expression(f"SELECT id FROM {table} WHERE {column} IN ({{slots}})", formatter)
        p = expression(f"SELECT id FROM {table} WHERE {column} IN ({{literals}})", formatter)
        a = shared + f"query = {q}\nreturn db.execute(query, tuple(items)).fetchall()"
        b = shared + f"query = {p}\nreturn db.execute(query, ()).fetchall()"
        matched, dynamic = True, True
        mechanism = "A generates only placeholder syntax and binds each list item; B builds SQL literals from the items."
    elif family in {"cte_predicate", "join_predicate", "nested_predicate", "aggregate_having", "case_predicate"}:
        if family == "cte_predicate":
            q = f"WITH selection AS (SELECT id FROM {table} WHERE {column} = ?) SELECT id FROM selection"
        elif family == "join_predicate":
            q = f"SELECT r.id FROM {table} AS r JOIN {table} AS p ON p.id = r.id WHERE r.{column} = ?"
        elif family == "nested_predicate":
            q = f"SELECT id FROM {table} WHERE id IN (SELECT id FROM {table} WHERE {column} = ?)"
        elif family == "aggregate_having":
            q = f"SELECT {other}, COUNT(*) FROM {table} GROUP BY {other} HAVING {other} = ?"
            normal = [["open"], ["missing"]]
        else:
            q = f"SELECT id FROM {table} WHERE CASE WHEN {column} = ? THEN 1 ELSE 0 END = 1"
        p = expression(q.replace("?", "'{value}'"), formatter)
        a = f"query = {q!r}\nreturn db.execute(query, (value,)).fetchall()"
        b = f"query = {p}\nreturn db.execute(query).fetchall()"
    elif family == "optional_filter":
        args, normal, witness = ["value", "other"], [["alpha", "open"], ["", "closed"]], [PREDICATE, "open"]
        suffix = f"else:\n    query = {f'SELECT id FROM {table} WHERE {other} = ?'!r}\n    parameters = (other,)\nreturn db.execute(query, parameters).fetchall()"
        a = f"if value:\n    query = {bound!r}\n    parameters = (value,)\n" + suffix
        b = f"if value:\n    query = {rendered}\n    parameters = ()\n" + suffix
        mechanism = "The nonempty-value branch differs; the empty-value branch binds the other value in both conditions."
    elif family == "insert_value":
        normal, witness = [["gamma"], ["missing"]], ["alpha'), ('extra"]
        q = f"INSERT INTO {table} ({column}) VALUES (?)"
        p = expression(q.replace("?", "'{value}'"), formatter)
        a = f"query = {q!r}\nreturn db.execute(query, (value,)).rowcount"
        b = f"query = {p}\nreturn db.execute(query).rowcount"
        mechanism = "A inserts one literal value; B's value can change the VALUES syntax and insert an additional row."
    elif family == "insert_partial_binding":
        args, normal = ["value", "other"], [["gamma", "open"], ["missing", "closed"]]
        q = f"INSERT INTO {table} ({column}, {other}) VALUES (?, ?)"
        p = expression(f"INSERT INTO {table} ({column}, {other}) VALUES (?, '{{other}}')", formatter)
        # Use a SQLite expression injection that preserves the single placeholder.
        witness = ["gamma", "open' || (SELECT {column} FROM {table} WHERE id = 2) || '".format(column=column, table=table)]
        a = f"query = {q!r}\nreturn db.execute(query, (value, other)).rowcount"
        b = f"query = {p}\nreturn db.execute(query, (value,)).rowcount"
        mechanism = "One value stays bound on B; the other is parsed as a SQL expression and can read fixture data instead of remaining literal text."
    elif family in {"update_predicate", "delete_predicate", "batch_update", "update_set_value", "named_batch_preview"}:
        if family == "delete_predicate":
            q = f"DELETE FROM {table} WHERE {column} = ?"
            p = expression(q.replace("?", "'{value}'"), formatter)
            a = f"query = {q!r}\nreturn db.execute(query, (value,)).rowcount"
            b = f"query = {p}\nreturn db.execute(query).rowcount"
        else:
            args, normal, witness = ["value", "other"], [["alpha", "revised"], ["missing", "revised"]], [PREDICATE, "revised"]
            q = f"UPDATE {table} SET {note} = ? WHERE {column} = ?"
            p = expression(f"UPDATE {table} SET {note} = ? WHERE {column} = '{{value}}'", formatter)
            if family == "update_set_value":
                p = expression(f"UPDATE {table} SET {note} = '{{other}}' WHERE {column} = ?", formatter)
                witness = ["alpha", f"changed', {other} = 'closed"]
                a = f"query = {q!r}\nreturn db.execute(query, (other, value)).rowcount"
                b = f"query = {p}\nreturn db.execute(query, (value,)).rowcount"
                mechanism = "B binds the WHERE value but parses the SET value as SQL, allowing another column assignment."
            elif family == "batch_update":
                args = ["value", "first", "second"]
                normal, witness = [["alpha", "first", "last"], ["missing", "first", "last"]], [PREDICATE, "first", "last"]
                a = f"query = {q!r}\nrows = ((first, value), (second, value))\nreturn db.executemany(query, rows).rowcount"
                b = f"query = {p}\nrows = ((first,), (second,))\nreturn db.executemany(query, rows).rowcount"
            elif family == "named_batch_preview":
                q = f"UPDATE {table} SET {note} = :note WHERE {column} = :item"
                p = expression(f"UPDATE {table} SET {note} = :note WHERE {column} = '{{value}}'", formatter)
                shared = f"statement = {q!r}\npreview = {p}\nrows = ({{'note': other, 'item': value}}, {{'note': other + '2', 'item': value}})\n"
                a = shared + "return db.executemany(statement, rows).rowcount"
                b = shared + "return db.executemany(preview, rows).rowcount"
                matched = True
            else:
                a = f"query = {q!r}\nreturn db.execute(query, (other, value)).rowcount"
                b = f"query = {p}\nreturn db.execute(query, (other,)).rowcount"
            if family != "update_set_value":
                mechanism = "A binds the predicate on every execution; B binds updated data but interpolates the predicate, permitting additional fixture rows to be updated."
        if family == "delete_predicate":
            mechanism = "A treats the predicate value as data; B can parse it as an always-true predicate and delete additional fixture rows."
    elif family == "cursor_loop":
        args, normal, witness = ["value", "other"], [["alpha", "beta"], ["missing", "unknown"]], [PREDICATE, "beta"]
        shared = "cursor = db.cursor()\nresults = []\nfor item in (value, other):\n"
        a = shared + f"    query = {bound!r}\n    cursor.execute(query, (item,))\n    results.extend(cursor.fetchall())\nreturn results"
        p = expression(literal.replace("{value}", "{item}"), formatter)
        b = shared + f"    query = {p}\n    cursor.execute(query)\n    results.extend(cursor.fetchall())\nreturn results"
    elif family in {"union_partial_binding", "cte_two_paths", "join_list_partial_binding"}:
        args, normal, witness = ["value", "other"], [["alpha", "open"], ["missing", "unknown"]], ["alpha", PREDICATE]
        if family == "union_partial_binding":
            q = f"SELECT id FROM {table} WHERE {column} = ? UNION SELECT id FROM {table} WHERE {other} = ?"
        elif family == "cte_two_paths":
            q = f"WITH selection AS (SELECT id FROM {table} WHERE {column} = ?) SELECT id FROM {table} WHERE id IN (SELECT id FROM selection) OR {other} = ?"
        else:
            q = f"SELECT r.id FROM {table} AS r JOIN {table} AS p ON p.id = r.id WHERE r.{column} IN (?, ?)"
            normal = [["alpha", "beta"], ["missing", "unknown"]]
            witness = ["alpha", "missing') OR ('1' LIKE '1"]
        first, last = q.rsplit("?", 1)
        p = expression(first + "'{other}'" + last, formatter)
        shared = f"statement = {q!r}\npreview = {p}\n"
        a = shared + "return db.execute(statement, (value, other)).fetchall()"
        b = shared + "return db.execute(preview, (value,)).fetchall()"
        matched = True
        mechanism = "A binds both paths or list members; B keeps one placeholder but places the other request value into SQL syntax."
    else:
        raise ValueError(f"Unknown template: {family}")

    schema = {table: [column, other, note, "amount"]}
    if family == "mapped_identifier":
        schema[table + "_copy"] = schema[table]
    return {
        "sources": {"A": source(a, args, variant), "B": source(b, args, variant)},
        "schema": schema,
        "normal_inputs": normal,
        "witness_inputs": witness,
        "quote_inputs": ["O'Reilly" if v not in {"current", "archive"} else v for v in normal[0]],
        "reference_mechanism": mechanism,
        "matched_control": matched,
        "dynamic_structure": dynamic,
        "formatting": formatter,
    }


def legacy_spec(split, stem):
    texts = {s: (DATASET / split / f"{stem}_{s}.txt").read_text(encoding="utf-8") for s in ("A", "B")}
    body = code(texts["A"])
    tree = ast.parse(body)
    arguments = [a.arg for a in tree.body[0].args.args][1:]
    if stem == "allowlisted_identifier":
        schema = {"vendors": ["name"], "partners": ["name"]}
        normal, witness, quotes = [["vendors", "alpha"], ["partners", "missing"]], ["vendors", PREDICATE], ["vendors", "O'Reilly"]
    else:
        table = re.search(r"FROM (\w+)", body)[1]
        fields = re.findall(r"(?:WHERE|AND|OR) (\w+) (?:=|LIKE)", body)
        schema = {table: list(dict.fromkeys(fields))}
        normal = [["alpha"] * len(arguments), ["missing"] * len(arguments)]
        witness, quotes = [PREDICATE] * len(arguments), ["O'Reilly"] * len(arguments)
        if stem == "two_values":
            normal, witness, quotes = [["alpha", "open"], ["missing", "open"]], ["alpha", PREDICATE], ["alpha", "O'Reilly"]
    group = stem
    if split == "train":
        index = LEGACY_STEMS[split].index(stem)
        group = "minimal" if index < 6 else "preview_direct" if index < 9 else "preview_starred"
    return {
        "sources": {s: code(texts[s]) for s in texts},
        "schema": schema,
        "normal_inputs": normal,
        "witness_inputs": witness,
        "quote_inputs": quotes,
        "template_family": "legacy_" + group,
        "template_group": "legacy:" + split + ":" + group,
        "formatting": "legacy",
        "matched_control": "preview =" in body,
        "dynamic_structure": stem == "allowlisted_identifier",
        "reference_mechanism": "A binds the supplied predicate values. B interpolates at least one value into the executed SQLite query; unrelated preview text or binding of another value does not protect it.",
    }


def build_dataset():
    # Refuse to discard an unexpected manual extension of these manifests.
    for split, filename in MANIFESTS.items():
        raw = json.loads((DATASET / filename).read_text(encoding="utf-8"))
        accepted = {f"{split}_{stem}" for stem in LEGACY_STEMS[split]}
        accepted.update(f"{split}_{family}_{v + 1:02d}" for family in {"train": TRAIN_FAMILIES, "validation": VALIDATION_FAMILIES, "test": TEST_FAMILIES}[split] for v in range(4 if split == "train" else 2))
        if any(p["id"] not in accepted for p in raw["pairs"]):
            raise ValueError("Manifest contains an additional manually authored pair; preserve it before rebuilding.")

    # Retire only previously generated assets whose saved hashes still match.
    # An edited or independently added file is never removed by this selection.
    retained_train_ids = {f"train_{stem}" for stem in LEGACY_STEMS["train"]}
    retained_train_ids.update(f"train_{family}_{variant + 1:02d}" for family, variant in TRAIN_SELECTION.items())
    retired_assets = []
    metadata_path = DATASET / "dataset_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for pair_id, case in previous["cases"].items():
            if case["split"] != "train" or pair_id in retained_train_ids:
                continue
            if pair_id in {f"train_{stem}" for stem in LEGACY_STEMS["train"]}:
                raise ValueError("The compact selection must preserve legacy inputs.")
            for side in ("A", "B"):
                path = (DATASET / case["files"][side]).resolve()
                if not path.is_relative_to((DATASET / "train").resolve()):
                    raise ValueError("Retired assets must remain inside the train directory.")
                if path.exists():
                    if digest(path.read_bytes()) != case["sha256"][side]:
                        raise ValueError(f"Preserve the manually edited asset before rebuilding: {path}")
                    retired_assets.append(path)

    ledger = {"dataset_revision": REVISION, "provenance": "Synthetic repository-authored fixtures; no independent real-world cases are claimed.", "references": ["https://cwe.mitre.org/data/definitions/89.html", "https://docs.python.org/3.12/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries"], "cases": {}}
    domains = iter(DOMAINS)
    for split, families in (("train", TRAIN_FAMILIES), ("validation", VALIDATION_FAMILIES), ("test", TEST_FAMILIES)):
        pairs = []

        def register(stem, spec, legacy=False):
            pair_id = f"{split}_{stem}"
            pair = {"id": pair_id, "template_group": spec["template_group"]}
            spec["sources"] = single_line_sources(spec["sources"])
            spec["matched_control"] = True
            texts = {}
            for side in ("A", "B"):
                relative = f"{split}/{stem}_{side}.txt"
                path = DATASET / relative
                context = CONTEXT + ("\n" + TASK if split != "train" else "")
                texts[side] = context + "\n\n```python\n" + spec["sources"][side].rstrip() + "\n```\n"
                path.write_text(texts[side], encoding="utf-8")
                pair[side] = {"text_file": relative, "image": ""}
            metadata = {k: v for k, v in spec.items() if k != "sources"}
            metadata.update({"split": split, "legacy_origin": legacy, "contrast": pair_contrast(texts["A"], texts["B"]), "files": {s: pair[s]["text_file"] for s in ("A", "B")}, "sha256": {s: digest((DATASET / pair[s]["text_file"]).read_bytes()) for s in ("A", "B")}})
            ledger["cases"][pair_id] = metadata
            pairs.append(pair)

        for stem in LEGACY_STEMS[split]:
            register(stem, legacy_spec(split, stem), legacy=True)
        for family_index, family in enumerate(families):
            domain = next(domains)
            variants = ([TRAIN_SELECTION[family]] if family in TRAIN_SELECTION else []) if split == "train" else range(2)
            for variant in variants:
                suffix = ("", "_archive", "_staging", "_history")[variant]
                table = domain + suffix
                column = ("code", "reference", "tag", "owner")[variant]
                other = ("status", "region", "category", "channel")[variant]
                note = ("note", "label", "description", "memo")[variant]
                formatter = FORMATS[variant if split == "train" else (family_index + variant) % 4]
                spec = make_spec(family, table, column, other, note, formatter, variant)
                spec.update({"template_family": family, "template_group": f"v3:{split}:{family}"})
                for body in spec["sources"].values():
                    ast.parse(body)
                register(f"{family}_{variant + 1:02d}", spec)

        manifest = {
            "version": 1,
            "name": "SQL injection · Code flow" + (f" · {split}" if split != "train" else ""),
            "asset_root": "sql_injection",
            "split": split,
            "labels": {"A": "Executed query binds request values", "B": "Executed query interpolates a request value"},
            "description": "20 code-only calibration pairs with exactly one changed line and byte-identical context before and after it. Preserves the 12 original pair IDs plus eight selected mechanisms; no reference answers in model inputs." if split == "train" else "20 reserved code-flow pairs with exactly one changed line and byte-identical surrounding text. Select settings on validation only; evaluate frozen settings on test. Contrast locations and reference mechanisms are evaluator-only metadata.",
            "pairs": pairs,
            "dataset_revision": REVISION,
            "metadata_file": "dataset_metadata.json",
        }
        write_json(DATASET / MANIFESTS[split], manifest)
    write_json(DATASET / "dataset_metadata.json", ledger)
    for path in retired_assets:
        path.unlink()


class ObservedCursor:
    def __init__(self, owner, cursor):
        self.owner, self.cursor = owner, cursor

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def execute(self, query, parameters=()):
        self.owner.calls.append(query)
        self.cursor.execute(query, parameters)
        return self

    def executemany(self, query, parameters):
        self.owner.calls.append(query)
        self.cursor.executemany(query, parameters)
        return self

    def fetchall(self):
        rows = self.cursor.fetchall()
        self.owner.results.append(rows)
        return rows


class ObservedConnection:
    def __init__(self, connection):
        self.connection, self.calls, self.results = connection, [], []

    def cursor(self):
        return ObservedCursor(self, self.connection.cursor())

    def execute(self, query, parameters=()):
        return self.cursor().execute(query, parameters)

    def executemany(self, query, parameters):
        return self.cursor().executemany(query, parameters)


def run_fixture(body, case, inputs):
    """Execute only audited repository fixtures against fresh in-memory tables.

    This is not a sandbox for uploaded/generated/untrusted Python. The dashboard
    remains a static consumer; this helper is exclusively an offline fixture test.
    """
    tree = ast.parse(body)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("A fixture needs one function and no top-level statements.")
    forbidden = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncFunctionDef, ast.ClassDef, ast.Global, ast.Nonlocal, ast.While)
    if any(isinstance(n, forbidden) for n in ast.walk(tree)):
        raise ValueError("Unsupported statement in repository fixture.")
    if any(isinstance(n, ast.Attribute) and n.attr.startswith("_") for n in ast.walk(tree)):
        raise ValueError("Private attributes are not allowed in fixture code.")
    namespace = {"__builtins__": {"tuple": tuple, "ValueError": ValueError}}
    exec(compile(tree, "<repository-sql-fixture>", "exec"), namespace)
    connection = sqlite3.connect(":memory:")
    try:
        for table, fields in case["schema"].items():
            if not all(re.fullmatch(r"[a-z][a-z0-9_]*", s) for s in [table, *fields]):
                raise ValueError("Fixture SQL identifiers must be simple lowercase names.")
            columns = ", ".join(f'"{name}" TEXT' for name in fields)
            connection.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY, {columns})')
            rows = []
            for index, first, second in ((1, "alpha", "open"), (2, "beta", "closed")):
                values = [first if position == 0 or name == "alias" else second if position == 1 else "initial" for position, name in enumerate(fields)]
                rows.append((index, *values))
            connection.executemany(f'INSERT INTO "{table}" VALUES ({", ".join("?" for _ in range(1 + len(fields)))})', rows)

        def authorize(action, table, field, *unused):
            if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}:
                return sqlite3.SQLITE_OK
            if action in {sqlite3.SQLITE_READ, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE} and table in case["schema"]:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        connection.set_progress_handler(lambda: 1, 10000)
        observed = ObservedConnection(connection)
        function = namespace[tree.body[0].name]
        function(observed, *inputs)
        snapshots = {table: connection.execute(f'SELECT * FROM "{table}" ORDER BY id').fetchall() for table in case["schema"]}
        return {"reads": observed.results, "tables": snapshots}, observed.calls
    finally:
        connection.close()


def audit_dataset(directory=DATASET):
    directory = Path(directory)
    ledger = json.loads((directory / "dataset_metadata.json").read_text(encoding="utf-8"))
    if ledger["dataset_revision"] != REVISION:
        raise ValueError("Unexpected dataset revision.")
    seen_files, seen_ids, seen_codes = set(), set(), {}
    splits, groups, formats = Counter(), {}, Counter()
    entries, checks, failures = [], Counter(), []
    # The region selector itself has no model or third-party dependencies.
    sys.path.insert(0, str(ROOT))
    from research.code_regions import select_regions

    for split, filename in MANIFESTS.items():
        manifest = json.loads((directory / filename).read_text(encoding="utf-8"))
        groups[split] = set()
        if manifest["split"] != split or manifest["dataset_revision"] != REVISION:
            raise ValueError(f"Wrong split/revision in {filename}.")
        for pair in manifest["pairs"]:
            pair_id = pair["id"]
            if pair_id in seen_ids:
                raise ValueError(f"Repeated pair ID: {pair_id}")
            seen_ids.add(pair_id)
            case = ledger["cases"][pair_id]
            if case["split"] != split or case["template_group"] != pair["template_group"]:
                raise ValueError(f"Metadata disagrees with {pair_id}.")
            groups[split].add(case["template_group"])
            formats[case["formatting"]] += 1
            bodies, texts = {}, {}
            for side in ("A", "B"):
                relative = pair[side]["text_file"]
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory.resolve()) or relative in seen_files:
                    raise ValueError(f"Repeated or unconfined asset: {relative}")
                seen_files.add(relative)
                if case["files"][side] != relative or case["sha256"][side] != digest(path.read_bytes()):
                    raise ValueError(f"Stale metadata hash: {relative}")
                texts[side] = path.read_text(encoding="utf-8")
                bodies[side] = code(texts[side])
                body_hash = digest(ast.dump(ast.parse(bodies[side]), include_attributes=False).encode())
                if body_hash in seen_codes:
                    raise ValueError(f"Duplicate code: {relative}, {seen_codes[body_hash]}")
                seen_codes[body_hash] = relative
                expected_context = CONTEXT + ("\n" + TASK if split != "train" else "")
                if texts[side].split("\n\n```python")[0] != expected_context:
                    raise ValueError(f"Unexpected context or instruction: {relative}")
                if re.search(r"sql.?injection|CWE-89|vulnerab|\b(?:safe|unsafe|benign|malicious|YES|NO)\b", texts[side], re.I):
                    raise ValueError(f"Answer/security cue in model input: {relative}")
                for scope in ("code", "sql_query", "sql_call", "sql_flow"):
                    if not select_regions(texts[side], scope):
                        raise ValueError(f"Empty capture region: {relative}/{scope}")
                    checks["capture_regions"] += 1
            if texts["A"] == texts["B"]:
                raise ValueError(f"Identical conditions: {pair_id}")
            contrast = pair_contrast(texts["A"], texts["B"])
            if contrast != case["contrast"]:
                raise ValueError(f"Stale contrast metadata: {pair_id}")
            checks["single_line_contrasts"] += 1
            result = {"pair_id": pair_id, "split": split, "template_group": case["template_group"], "matched_control": case["matched_control"], "contrast": contrast}
            try:
                for inputs in case["normal_inputs"]:
                    a, _ = run_fixture(bodies["A"], case, inputs)
                    b, _ = run_fixture(bodies["B"], case, inputs)
                    if a != b:
                        raise ValueError("A/B database behavior differs on ordinary inputs.")
                    checks["ordinary_equivalence"] += 1
                # Apostrophes must be accepted on A. A syntax error on B is not
                # accepted as the successful behavioral injection witness below.
                _, quote_calls = run_fixture(bodies["A"], case, case["quote_inputs"])
                if any("O'Reilly" in query for query in quote_calls):
                    raise ValueError("A puts an apostrophe-bearing request value into SQL text.")
                checks["A_quote_inputs"] += 1
                a, a_calls = run_fixture(bodies["A"], case, case["witness_inputs"])
                b, b_calls = run_fixture(bodies["B"], case, case["witness_inputs"])
                if a == b:
                    raise ValueError("Injection witness has no observable database effect.")
                if not a_calls or not b_calls:
                    raise ValueError("Witness did not reach a database call.")
                if any("'1' LIKE '1" in query or "O'Reilly" in query for query in a_calls):
                    raise ValueError("A executes witness syntax as SQL text.")
                checks["successful_B_witnesses"] += 1
                result["status"] = "passed"
            except (sqlite3.Error, ValueError, TypeError, KeyError) as error:
                result.update({"status": "failed", "error": str(error)})
                failures.append(result)
            entries.append(result)
            splits[split] += 1

    if set(ledger["cases"]) != seen_ids:
        raise ValueError("Metadata contains missing or extra cases.")
    if splits != PAIR_COUNTS:
        raise ValueError(f"Unexpected split sizes: {dict(splits)}")
    tracked_assets = {str(p.relative_to(directory)) for split in MANIFESTS for p in (directory / split).glob("*.txt")}
    if tracked_assets != seen_files:
        raise ValueError("Unregistered text assets exist in dataset splits.")
    report = {
        "dataset_revision": REVISION,
        "status": "passed" if not failures else "failed",
        "pairs_by_split": dict(splits),
        "total_pairs": sum(splits.values()),
        "total_prompt_assets": len(seen_files),
        "template_groups_by_split": {s: len(g) for s, g in groups.items()},
        "formatting_pairs": dict(formats),
        "matched_control_pairs": sum(e["matched_control"] for e in entries),
        "legacy_pair_ids_preserved": sum(c["legacy_origin"] for c in ledger["cases"].values()),
        "contrast_kinds": dict(Counter(e["contrast"]["kind"] for e in entries)),
        "checks": dict(checks),
        "sqlite_version": sqlite3.sqlite_version,
        "dataset_sha256": digest(json.dumps({pair_id: case["sha256"] for pair_id, case in sorted(ledger["cases"].items())}, sort_keys=True).encode()),
        "method": "Exactly one changed physical line per pair, with byte-identical prefix/suffix. Fresh in-memory SQLite fixtures verify ordinary-input equivalence, A apostrophe handling, and successful A/B witness executions with different query results or table contents. No model inference.",
        "limitations": ["Synthetic variants share templates; pair count is not the number of independent real-world mechanisms.", "Mechanism families overlap across splits; only examples and newly added template groups are held out.", "A successful witness verifies a local execution path, not the security of an entire application or the model's reasoning."],
        "pairs": entries,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "audit"))
    parser.add_argument("--report", type=Path, help="Write the full audit as JSON.")
    arguments = parser.parse_args()
    if arguments.command == "build":
        build_dataset()
    report = audit_dataset()
    if arguments.report:
        write_json(arguments.report, report)
    summary = {k: v for k, v in report.items() if k != "pairs"}
    if report["status"] != "passed":
        summary["failures"] = [p for p in report["pairs"] if p["status"] != "passed"]
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
