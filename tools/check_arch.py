#!/usr/bin/env python3
"""Architecture linter — enforces the hexagonal dependency rule by reading imports.

Rings, inner -> outer. Each ring may import ONLY itself + inner rings:

    domain   ->  (nothing internal)
    ports    ->  domain
    adapters ->  domain, ports
    app      ->  domain, ports, adapters      (the composition root)

Extra rule: the inner rings (domain, ports) may not import outside-world I/O
libraries (cv2, mediapipe, pynput, pyobjc, ...). Those belong only in adapters/.

Run:   .venv/bin/python tools/check_arch.py
Exit:  0 = clean, 1 = violations found.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "kinesis"

# a ring may import itself + every inner ring
ALLOWED = {
    "domain": {"domain"},
    "ports": {"domain", "ports"},
    "adapters": {"domain", "ports", "adapters"},
    "app": {"domain", "ports", "adapters", "app"},
}
RINGS = set(ALLOWED)

# outside-world libraries that must never appear in the inner rings
IO_LIBS = {
    "cv2", "mediapipe", "pynput", "Quartz", "AppKit", "Cocoa",
    "Foundation", "ApplicationServices", "objc",
}
IO_FORBIDDEN_IN = {"domain", "ports"}


def ring_of(path: pathlib.Path):
    """Ring a file belongs to, or None for files directly under kinesis/."""
    rel = path.relative_to(PKG).parts
    return rel[0] if len(rel) > 1 and rel[0] in RINGS else None


def imported_modules(tree, path):
    """Yield (absolute_module, lineno), resolving relative imports to kinesis.*."""
    pkg = path.relative_to(ROOT).with_suffix("").parts[:-1]  # containing package
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against this file's package
                base = list(pkg)
                for _ in range(node.level - 1):
                    if base:
                        base.pop()
                parts = base + ([node.module] if node.module else [])
                yield ".".join(parts), node.lineno
            elif node.module:
                yield node.module, node.lineno


def check():
    violations = []
    for path in sorted(PKG.rglob("*.py")):
        ring = ring_of(path)
        if ring is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            violations.append((path, e.lineno or 0, f"syntax error: {e.msg}"))
            continue
        for mod, lineno in imported_modules(tree, path):
            parts = mod.split(".")
            top = parts[0]
            # rule 1: internal ring dependencies must point inward
            if top == "kinesis" and len(parts) > 1 and parts[1] in RINGS:
                target = parts[1]
                if target not in ALLOWED[ring]:
                    violations.append((path, lineno,
                        f"{ring}/ imports kinesis.{target}.*  "
                        f"({ring} may only import {sorted(ALLOWED[ring])})"))
            # rule 2: inner rings must not touch outside-world I/O libs
            if ring in IO_FORBIDDEN_IN and top in IO_LIBS:
                violations.append((path, lineno,
                    f"{ring}/ imports outside-world lib '{top}'  "
                    f"(only adapters/ may import I/O libraries)"))
    return violations


def main():
    violations = check()
    if not violations:
        print("architecture OK — all dependencies point inward")
        return 0
    print(f"architecture violations ({len(violations)}):\n")
    for path, lineno, msg in violations:
        print(f"  {path.relative_to(ROOT)}:{lineno}: {msg}")
    print("\nrule: domain <- ports <- adapters <- app   (arrows point inward)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
