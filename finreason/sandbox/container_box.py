"""Sandbox security policies, AST validation, and execution guardrails.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Set


class SecurityViolationError(Exception):
    """Raised when code violates sandbox security constraints."""
    pass


@dataclass
class SandboxSecurityPolicy:
    """Security policy configuration for sandboxed execution."""
    allowed_modules: Set[str] = field(
        default_factory=lambda: {
            "math",
            "duckdb",
            "pandas",
            "numpy",
            "decimal",
            "datetime",
            "statistics",
            "collections",
            "itertools",
            "re",
            "json",
        }
    )
    forbidden_calls: Set[str] = field(
        default_factory=lambda: {
            "eval",
            "exec",
            "compile",
            "input",
            "breakpoint",
            "exit",
            "quit",
        }
    )
    timeout_seconds: float = 15.0
    max_output_chars: int = 50000
    max_output_rows: int = 100
    allow_network: bool = False
    strict_ast_check: bool = True


class ASTSecurityValidator(ast.NodeVisitor):
    """AST visitor that checks for unauthorized imports, calls, and dunder hacking."""

    def __init__(self, policy: SandboxSecurityPolicy):
        self.policy = policy
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module not in self.policy.allowed_modules:
                self.violations.append(
                    f"Import of module '{alias.name}' is prohibited. "
                    f"Allowed modules: {sorted(self.policy.allowed_modules)}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module not in self.policy.allowed_modules:
                self.violations.append(
                    f"Import from module '{node.module}' is prohibited. "
                    f"Allowed modules: {sorted(self.policy.allowed_modules)}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.policy.forbidden_calls:
                self.violations.append(
                    f"Calling '{node.func.id}()' is prohibited in the sandbox."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Prevent access to dangerous internal attributes like __subclasses__, __bases__, __globals__
        if node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr in {
                "__subclasses__",
                "__globals__",
                "__bases__",
                "__mro__",
                "__code__",
                "__closure__",
                "__import__",
            }:
                self.violations.append(
                    f"Accessing restricted attribute '{node.attr}' is prohibited."
                )
        self.generic_visit(node)


def validate_code(code: str, policy: SandboxSecurityPolicy) -> None:
    """Validate python code against the security policy using AST inspection."""
    if not policy.strict_ast_check:
        return

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityViolationError(f"Syntax error during code validation: {e}") from e

    validator = ASTSecurityValidator(policy)
    validator.visit(tree)

    if validator.violations:
        raise SecurityViolationError("; ".join(validator.violations))


def run_in_subprocess(
    code: str,
    tables: dict,
    context: dict,
    policy: SandboxSecurityPolicy,
) -> dict:
    """Execute code in a separate OS process with hard timeout termination."""
    import os
    import pickle
    import subprocess
    import sys
    import tempfile
    import time

    start_time = time.perf_counter()

    # Pre-validate AST before spawning process
    try:
        validate_code(code, policy)
    except SecurityViolationError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": "",
            "execution_time_ms": elapsed_ms,
            "error": f"SecurityViolation: {str(e)}",
            "tables": [],
            "updated_tables": {},
        }


    with tempfile.NamedTemporaryFile(delete=False, suffix=".in.pkl") as f_in, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".out.pkl") as f_out:
        in_path = f_in.name
        out_path = f_out.name
        pickle.dump(
            {
                "code": code,
                "tables": tables,
                "context": context,
                "policy": policy,
            },
            f_in,
        )

    try:
        cmd = [sys.executable, "-m", "finreason.sandbox.worker", in_path, out_path]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            _, stderr_data = proc.communicate(timeout=policy.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "result": None,
                "stdout": "",
                "stderr": "",
                "execution_time_ms": elapsed_ms,
                "error": f"Python execution timed out after {policy.timeout_seconds}s",
                "tables": [],
                "updated_tables": {},
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as f:
                res = pickle.load(f)
            return res

        err_msg = stderr_data.decode("utf-8", errors="replace") if stderr_data else "Unknown error"
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": err_msg,
            "execution_time_ms": elapsed_ms,
            "error": f"Process exited with code {proc.returncode}: {err_msg}",
            "tables": [],
            "updated_tables": {},
        }

    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

