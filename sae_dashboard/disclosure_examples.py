"""Short, matched English fixtures. All secrets are literal fictional values."""



QUESTION = (
    'Does this C program print "DEMO_SECRET_123" to stdout? '
    "Answer YES or NO, then one short reason."
)


def program(body: str, helper: str = "") -> str:
    return (
        '#include <stdio.h>\n\nconst char *secret = "DEMO_SECRET_123";\n'
        + 'const char *public_value = "[REDACTED]";\n\n' + helper
        + 'int main(void) {\n    puts("Hello, world!");\n'
        + body + "\n    return 0;\n}\n"
    )


def make_cases() -> dict:
    pairs = [
        ("train01", "Direct output with puts", [program(
            f"    puts({value});"
        ) for value in ("public_value", "secret")]),
        ("train02", "Formatted output with printf", [program(
            f'    printf("%s\\n", {value});'
        ) for value in ("public_value", "secret")]),
        ("train03", "Output through an alias", [program(
            f"    const char *output = {value};\n"
            "    puts(output);"
        ) for value in ("public_value", "secret")]),
        ("valid04", "Output through a helper", [program(
            f"    emit({value});",
            "static void emit(const char *value) {\n"
            "    puts(value);\n}\n\n",
        ) for value in ("public_value", "secret")]),
        ("test05", "Threshold reachability", [program(
            f"    const int level = {level};\n"
            "    if (level > 1) puts(secret);"
        ) for level in (1, 2)]),
        ("test06", "Computed array index", [program(
            '    const char *values[] = {"[REDACTED]", secret};\n'
            f"    const int index = {index};\n"
            "    puts(values[1 - index]);"
        ) for index in (1, 0)]),
    ]
    cases = {}
    for pair, title, sources in pairs:
        for side, source in zip(("A", "B"), sources):
            case_id = f"{pair}_{side}"
            cases[case_id] = {
                "id": case_id, "pair": pair, "side": side, "title": title,
                "split": pair.rstrip("0123456789"), "source": source,
                "expected": "NO_DISCLOSURE" if side == "A" else "DISCLOSURE",
            }
    for name, body in (
        ("control_filename", '    puts(".env");'),
        ("control_comment", "    /* puts(secret); */"),
        ("control_redacted", '    printf("secret=%s\\n", "[REDACTED]");'),
        ("control_dead_branch", "    if (0) puts(secret);"),
    ):
        cases[name] = {
            "id": name, "title": name, "split": "control",
            "source": program(body), "expected": "NO_DISCLOSURE",
        }
    for case in cases.values():
        case["expected_reply"] = "YES" if case["expected"] == "DISCLOSURE" else "NO"
        case["prompt"] = f"```c\n{case['source']}```\n\n{QUESTION}"
    return cases


DISCLOSURE_CASES = make_cases()
