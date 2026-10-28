"""Episodic Memory (专案因果侦查日志与 Reflexion 回滚缓冲区).
Tracks multi-turn investigative actions, observation traces, and counterfactual reflections.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
from pydantic import BaseModel, Field


class InvestigationStep(BaseModel):
    """A single step in a financial forensic investigation trajectory."""
    step_id: int = Field(default=1, description="Sequential step index")
    case_id: str = Field(description="Associated case identifier")
    action_type: str = Field(default="audit_action", description="Action name (e.g. sandbox_calc, footnote_dereference, hypothesis_test)")
    action_payload: str = Field(default="", description="Exact code, query, or command executed")
    observation: str = Field(default="", description="Observed outcome or error")
    status: str = Field(default="confirmed", description="Outcome status: confirmed, refuted, inconclusive, error")
    hypothesis: Optional[str] = Field(default=None, description="Hypothesis being evaluated if applicable")
    reflection: Optional[str] = Field(default=None, description="Reflexion self-correction note on why it failed or succeeded")
    timestamp: float = Field(default_factory=time.time)


class EpisodicMemory:
    """Persistent DuckDB storage for investigation trajectories and Reflexion memory."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.con = duckdb.connect(database=db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        """Initialize the investigation episodes and reflexions tables."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS investigation_steps (
                case_id VARCHAR,
                step_id INTEGER,
                action_type VARCHAR,
                action_payload VARCHAR,
                observation VARCHAR,
                status VARCHAR,
                hypothesis VARCHAR,
                reflection VARCHAR,
                timestamp DOUBLE,
                PRIMARY KEY (case_id, step_id)
            );
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS case_reflections (
                case_id VARCHAR,
                reflection_id INTEGER,
                refuted_hypothesis VARCHAR,
                lesson_learned VARCHAR,
                timestamp DOUBLE
            );
        """)

    def close(self) -> None:
        if self.con:
            self.con.close()

    def log_step(self, step: InvestigationStep) -> None:
        """Record a single step into the episodic log."""
        self.con.execute("""
            INSERT OR REPLACE INTO investigation_steps
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            step.case_id,
            step.step_id,
            step.action_type,
            step.action_payload,
            step.observation,
            step.status,
            step.hypothesis or "",
            step.reflection or "",
            step.timestamp,
        ])

        # If this step was refuted or has a self-reflection, log it to reflections table
        if step.status == "refuted" or step.reflection:
            self.add_reflection(
                case_id=step.case_id,
                refuted_hypothesis=step.hypothesis or "Action Failed",
                lesson_learned=step.reflection or f"Action '{step.action_type}' yielded negative evidence: {step.observation[:200]}",
            )

    def record_step(self, step: InvestigationStep) -> None:
        """Alias for log_step to record an investigation step."""
        self.log_step(step)


    def add_reflection(self, case_id: str, refuted_hypothesis: str, lesson_learned: str) -> None:
        """Add an explicit Reflexion lesson to avoid repeating false avenues."""
        next_id = self.con.execute(
            "SELECT COALESCE(MAX(reflection_id), 0) + 1 FROM case_reflections WHERE case_id = ?",
            [case_id]
        ).fetchone()[0]

        self.con.execute("""
            INSERT INTO case_reflections VALUES (?, ?, ?, ?, ?)
        """, [case_id, next_id, refuted_hypothesis, lesson_learned, time.time()])

    def get_trajectory(self, case_id: str) -> List[InvestigationStep]:
        """Fetch all chronological steps for a given case."""
        df = self.con.execute("""
            SELECT step_id, case_id, action_type, action_payload, observation, status, hypothesis, reflection, timestamp
            FROM investigation_steps
            WHERE case_id = ?
            ORDER BY step_id ASC
        """, [case_id]).fetchdf()

        steps = []
        for _, row in df.iterrows():
            steps.append(InvestigationStep(
                step_id=int(row["step_id"]),
                case_id=str(row["case_id"]),
                action_type=str(row["action_type"]),
                action_payload=str(row["action_payload"]),
                observation=str(row["observation"]),
                status=str(row["status"]),
                hypothesis=str(row["hypothesis"]) if row["hypothesis"] else None,
                reflection=str(row["reflection"]) if row["reflection"] else None,
                timestamp=float(row["timestamp"]),
            ))
        return steps

    def get_reflections(self, case_id: str, limit: int = 5) -> List[str]:
        """Fetch recent reflexions to inject into the agent prompt, preventing dead ends."""
        df = self.con.execute("""
            SELECT refuted_hypothesis, lesson_learned
            FROM case_reflections
            WHERE case_id = ?
            ORDER BY reflection_id DESC
            LIMIT ?
        """, [case_id, limit]).fetchdf()

        reflections = []
        for _, row in df.iterrows():
            reflections.append(f"【反思教训】被证伪假设: '{row['refuted_hypothesis']}' -> 经验总结: {row['lesson_learned']}")
        return reflections

    def get_refuted_hypotheses(self, case_id: str) -> List[str]:
        """Get all hypotheses that have already been disproven in this case."""
        df = self.con.execute("""
            SELECT DISTINCT refuted_hypothesis
            FROM case_reflections
            WHERE case_id = ? AND refuted_hypothesis != ''
        """, [case_id]).fetchdf()
        if "refuted_hypothesis" in df.columns:
            return df["refuted_hypothesis"].tolist()
        return []

    def format_reflection_prompt(self, case_id: str) -> str:
        """Format reflexions into an anti-repetition guidance block for LLM planning."""
        reflections = self.get_reflections(case_id)
        if not reflections:
            return ""

        return (
            "【历史调查反思与排障警示 (Reflexion Memory)】:\n"
            + "\n".join(f"  • {r}" for r in reflections)
            + "\n注意：请吸取上述反思教训，严禁重复探索已被证据推翻的死路假设！"
        )
