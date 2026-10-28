"""Deterministic execution sandbox powered by DuckDB and isolated Python environment.
Ensures zero arithmetic hallucination for financial metrics, ratios, and audit balancing.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
import concurrent.futures

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from finreason.sandbox.container_box import (
    SandboxSecurityPolicy,
    SecurityViolationError,
    run_in_subprocess,
    validate_code,
)



class ExecutionResult(BaseModel):
    """Result of sandboxed code or SQL execution."""
    success: bool = Field(description="Whether execution completed without error")
    result: Any = Field(default=None, description="Returned result or evaluated object")
    stdout: str = Field(default="", description="Captured standard output")
    stderr: str = Field(default="", description="Captured standard error or warning")
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    tables: List[str] = Field(default_factory=list, description="Tables currently in the database")

    def to_audit_summary(self) -> str:
        """Format a clear summary for agent reasoning logs."""
        if not self.success:
            return f"[EXECUTION FAILED] Error: {self.error}\nStderr: {self.stderr}"

        summary_parts = [f"[EXECUTION SUCCESS] ({self.execution_time_ms:.2f}ms)"]
        if self.stdout:
            summary_parts.append(f"Output:\n{self.stdout.strip()}")
        if self.result is not None:
            if isinstance(self.result, pd.DataFrame):
                summary_parts.append(f"Result DataFrame ({len(self.result)} rows):\n{self.result.to_string()}")
            else:
                summary_parts.append(f"Result: {self.result}")
        return "\n".join(summary_parts)


class CodeRunner:
    """Zero-arithmetic-hallucination executor combining DuckDB and isolated Python.

    Maintains an in-memory DuckDB database and allows:
    1. Registering financial tables (DataFrames, CSVs, dicts).
    2. Running deterministic SQL queries directly.
    3. Running validated Python code for multi-step financial calculations.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        policy: Optional[SandboxSecurityPolicy] = None,
    ):
        self.policy = policy or SandboxSecurityPolicy()
        self.con = duckdb.connect(database=db_path)
        # Configure DuckDB settings
        self.con.execute("SET preserve_insertion_order=true")

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        if self.con:
            self.con.close()

    def __enter__(self) -> CodeRunner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def list_tables(self) -> List[str]:
        """List all user tables/views in the DuckDB session."""
        df = self.con.execute("SHOW TABLES").fetchdf()
        if "name" in df.columns:
            return df["name"].tolist()
        return []

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Register a Pandas DataFrame as a DuckDB table."""
        self.con.register(f"_temp_{table_name}", df)
        self.con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _temp_{table_name}")
        self.con.unregister(f"_temp_{table_name}")

    def register_records(self, table_name: str, records: List[Dict[str, Any]]) -> None:
        """Register a list of row dicts as a DuckDB table."""
        df = pd.DataFrame(records)
        self.register_dataframe(table_name, df)

    def get_table_df(self, table_name: str) -> pd.DataFrame:
        """Fetch an existing table as a pandas DataFrame."""
        return self.con.execute(f"SELECT * FROM {table_name}").fetchdf()

    def execute_sql(self, sql: str) -> ExecutionResult:
        """Execute one or more SQL statements in DuckDB deterministically."""
        start_time = time.perf_counter()
        sql_clean = sql.strip()
        if not sql_clean:
            return ExecutionResult(
                success=False,
                error="Empty SQL query provided",
                execution_time_ms=0.0,
                tables=self.list_tables(),
            )

        try:
            # Check timeout via thread pool
            def _run():
                statements = [s.strip() for s in sql_clean.split(";") if s.strip()]
                res_df = None
                for stmt in statements:
                    cursor = self.con.execute(stmt)
                    # If this statement returns data (e.g. SELECT, PRAGMA)
                    if cursor.description is not None:
                        res_df = cursor.fetch_df_chunk(self.policy.max_output_rows)
                return res_df

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                res_df = future.result(timeout=self.policy.timeout_seconds)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            stdout_str = ""
            if res_df is not None:
                stdout_str = res_df.to_string(index=False)
                if len(stdout_str) > self.policy.max_output_chars:
                    stdout_str = stdout_str[: self.policy.max_output_chars] + "\n...[Output truncated]"

            return ExecutionResult(
                success=True,
                result=res_df,
                stdout=stdout_str,
                execution_time_ms=elapsed_ms,
                tables=self.list_tables(),
            )

        except concurrent.futures.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                error=f"SQL execution timed out after {self.policy.timeout_seconds}s",
                execution_time_ms=elapsed_ms,
                tables=self.list_tables(),
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=elapsed_ms,
                tables=self.list_tables(),
            )

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """General execution dispatcher (Python or SQL) for sandboxed evaluation."""
        stripped = code.strip()
        if stripped.upper().startswith(("SELECT ", "WITH ", "SHOW ", "DESCRIBE ", "CREATE ", "INSERT ", "PRAGMA ")):
            return self.execute_sql(code)
        return self.execute_python(code, context=context)

    def execute_python(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute python code in a sandboxed namespace with DuckDB and financial tools."""
        # 1. Gather all existing tables to provide to the isolated worker
        tables_dict = {}
        for tbl in self.list_tables():
            try:
                tables_dict[tbl] = self.get_table_df(tbl)
            except Exception:
                pass

        # 2. Run in isolated subprocess with hard-kill timeout
        res_dict = run_in_subprocess(
            code=code,
            tables=tables_dict,
            context=context or {},
            policy=self.policy,
        )

        # 3. Synchronize any newly created or modified tables back into this DuckDB session
        updated_tables = res_dict.get("updated_tables", {})
        for tbl_name, df in updated_tables.items():
            if isinstance(df, pd.DataFrame):
                try:
                    self.register_dataframe(tbl_name, df)
                except Exception:
                    pass

        return ExecutionResult(
            success=res_dict.get("success", False),
            result=res_dict.get("result", None),
            stdout=res_dict.get("stdout", ""),
            stderr=res_dict.get("stderr", ""),
            execution_time_ms=res_dict.get("execution_time_ms", 0.0),
            error=res_dict.get("error", None),
            tables=self.list_tables(),
        )

