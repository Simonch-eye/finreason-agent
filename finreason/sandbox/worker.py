"""Subprocess worker that executes Python code in complete OS process isolation.
"""

from __future__ import annotations

import contextlib
import io
import math
import pickle
import sys
import time
from decimal import Decimal
from typing import Any, Dict

import duckdb
import numpy as np
import pandas as pd

from finreason.sandbox.container_box import (
    SandboxSecurityPolicy,
    SecurityViolationError,
    validate_code,
)


def _safe_builtins() -> Dict[str, Any]:
    return {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
        "ZeroDivisionError": ZeroDivisionError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
    }


def _verify_equation(name: str, left: float, right: float, tolerance_pct: float = 0.001) -> Dict[str, Any]:
    diff = abs(left - right)
    base = max(abs(left), abs(right), 1e-9)
    rel_diff = diff / base
    balanced = rel_diff <= tolerance_pct
    return {
        "equation": name,
        "left": left,
        "right": right,
        "diff": diff,
        "relative_diff": rel_diff,
        "balanced": balanced,
    }


def _safe_ratio(num: float, denom: float) -> Any:
    if denom == 0 or math.isnan(denom):
        return None
    return num / denom


def run_worker(input_path: str, output_path: str) -> None:
    """Execute the sandboxed payload in isolation and write the serialized result."""
    start_time = time.perf_counter()

    try:
        with open(input_path, "rb") as f:
            payload = pickle.load(f)

        code: str = payload["code"]
        tables_dict: Dict[str, pd.DataFrame] = payload.get("tables", {})
        context: Dict[str, Any] = payload.get("context", {})
        policy: SandboxSecurityPolicy = payload.get("policy", SandboxSecurityPolicy())

        # 1. AST security check
        validate_code(code, policy)

        # 2. Setup DuckDB and register tables
        con = duckdb.connect(database=":memory:")
        con.execute("SET preserve_insertion_order=true")
        for tbl_name, df in tables_dict.items():
            con.register(f"_temp_{tbl_name}", df)
            con.execute(f"CREATE TABLE {tbl_name} AS SELECT * FROM _temp_{tbl_name}")
            con.unregister(f"_temp_{tbl_name}")

        env: Dict[str, Any] = {
            "__builtins__": _safe_builtins(),
            "duckdb": duckdb,
            "con": con,
            "db": con,
            "pd": pd,
            "np": np,
            "math": math,
            "Decimal": Decimal,
            "verify_equation": _verify_equation,
            "safe_ratio": _safe_ratio,
        }
        if context:
            for k, v in context.items():
                if k not in env:
                    env[k] = v

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            compiled = compile(code, "<sandbox>", "exec")
            exec(compiled, env)
            return_val = env.get("result", None)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Extract all tables currently in con
        all_tables_df = con.execute("SHOW TABLES").fetchdf()
        current_tables = all_tables_df["name"].tolist() if "name" in all_tables_df.columns else []
        updated_tables = {}
        for tbl in current_tables:
            updated_tables[tbl] = con.execute(f"SELECT * FROM {tbl}").fetchdf()

        stdout_val = stdout_buf.getvalue()
        if len(stdout_val) > policy.max_output_chars:
            stdout_val = stdout_val[: policy.max_output_chars] + "\n...[Output truncated]"

        response = {
            "success": True,
            "result": return_val,
            "stdout": stdout_val,
            "stderr": stderr_buf.getvalue(),
            "execution_time_ms": elapsed_ms,
            "error": None,
            "tables": current_tables,
            "updated_tables": updated_tables,
        }

    except SecurityViolationError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response = {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": "",
            "execution_time_ms": elapsed_ms,
            "error": f"SecurityViolation: {str(e)}",
            "tables": [],
            "updated_tables": {},
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response = {
            "success": False,
            "result": None,
            "stdout": stdout_buf.getvalue() if "stdout_buf" in locals() else "",
            "stderr": stderr_buf.getvalue() if "stderr_buf" in locals() else "",
            "execution_time_ms": elapsed_ms,
            "error": f"{type(e).__name__}: {str(e)}",
            "tables": [],
            "updated_tables": {},
        }

    with open(output_path, "wb") as f:
        pickle.dump(response, f)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_worker(sys.argv[1], sys.argv[2])
    else:
        sys.exit(1)
