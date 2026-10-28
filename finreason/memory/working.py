"""Working Memory (案头活跃草稿纸).
Maintains ephemeral case state, active investigative hypotheses, and scratchpad calculations.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkingMemorySnapshot(BaseModel):
    """Snapshot of working memory state for rollback and branching."""
    snapshot_id: str
    timestamp: float
    active_hypotheses: List[str]
    scratchpad_notes: List[str]
    active_variables: Dict[str, Any]


class WorkingMemory(BaseModel):
    """Auditor desktop working scratchpad (REPL context and ephemeral working variables)."""
    case_id: str = Field(default="case_default", description="Current investigation case identifier")
    target_company: str = Field(default="", description="Subject enterprise under audit")
    reporting_period: str = Field(default="", description="Fiscal period being audited")
    active_hypotheses: List[str] = Field(default_factory=list, description="Currently active forensic hypotheses")
    scratchpad_notes: List[str] = Field(default_factory=list, description="Ephemeral audit reasoning notes")
    active_variables: Dict[str, Any] = Field(default_factory=dict, description="Calculated values and sandbox outputs")
    token_budget: int = Field(default=4000, description="Maximum token allowance in prompt")
    snapshots: Dict[str, WorkingMemorySnapshot] = Field(default_factory=dict, description="Checkpoints for rollback")

    def add_hypothesis(self, hypothesis: str) -> None:
        """Add a new forensic hypothesis to test."""
        clean = hypothesis.strip()
        if clean and clean not in self.active_hypotheses:
            self.active_hypotheses.append(clean)

    def remove_hypothesis(self, hypothesis: str) -> None:
        """Remove or refute an active hypothesis."""
        if hypothesis in self.active_hypotheses:
            self.active_hypotheses.remove(hypothesis)

    def add_note(self, note: str) -> None:
        """Record an intermediate discovery or calculation summary on the scratchpad."""
        ts = time.strftime("%H:%M:%S")
        self.scratchpad_notes.append(f"[{ts}] {note.strip()}")

    def set_variable(self, key: str, value: Any) -> None:
        """Set a named variable (e.g. '2023_roe' = 0.1875)."""
        self.active_variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Retrieve a named variable from the scratchpad."""
        return self.active_variables.get(key, default)

    def create_checkpoint(self, name: Optional[str] = None) -> str:
        """Save a snapshot of the current working memory for branching or rollback."""
        snap_id = name or f"snap_{len(self.snapshots) + 1}_{int(time.time())}"
        snapshot = WorkingMemorySnapshot(
            snapshot_id=snap_id,
            timestamp=time.time(),
            active_hypotheses=copy.deepcopy(self.active_hypotheses),
            scratchpad_notes=copy.deepcopy(self.scratchpad_notes),
            active_variables=copy.deepcopy(self.active_variables),
        )
        self.snapshots[snap_id] = snapshot
        return snap_id

    def rollback(self, snapshot_id: str) -> bool:
        """Restore working memory state to a previous snapshot."""
        if snapshot_id not in self.snapshots:
            return False
        snap = self.snapshots[snapshot_id]
        self.active_hypotheses = copy.deepcopy(snap.active_hypotheses)
        self.scratchpad_notes = copy.deepcopy(snap.scratchpad_notes)
        self.active_variables = copy.deepcopy(snap.active_variables)
        return True

    def to_prompt_context(self) -> str:
        """Format the working scratchpad into a clean prompt context block."""
        parts = [
            f"=== 案头工作记忆 (Working Memory: {self.case_id}) ===",
            f"审计主体: {self.target_company or '未指定'} | 会计期间: {self.reporting_period or '未指定'}",
        ]

        if self.active_hypotheses:
            parts.append("\n【当前待验假设 (Active Hypotheses)】:")
            for i, h in enumerate(self.active_hypotheses, 1):
                parts.append(f"  {i}. {h}")

        if self.active_variables:
            parts.append("\n【已核算关键指标 (Active Variables)】:")
            for k, v in self.active_variables.items():
                if isinstance(v, float):
                    parts.append(f"  • {k}: {v:,.4f}")
                else:
                    parts.append(f"  • {k}: {v}")

        if self.scratchpad_notes:
            parts.append("\n【侦查草稿日志 (Scratchpad Notes)】:")
            for note in self.scratchpad_notes[-10:]:  # Keep most recent notes
                parts.append(f"  {note}")

        parts.append("================================================")
        full_text = "\n".join(parts)

        # Truncate if exceeding approximate char budget (1 token ~ 3.5 chars)
        max_chars = self.token_budget * 3
        if len(full_text) > max_chars:
            return full_text[:max_chars] + "\n...[草稿内容已达上限截断]"
        return full_text

    def clear(self) -> None:
        """Clear the working memory for a new investigation task."""
        self.active_hypotheses.clear()
        self.scratchpad_notes.clear()
        self.active_variables.clear()
        self.snapshots.clear()

    def reset(self, case_id: str = "case_default", target_company: str = "", reporting_period: str = "") -> None:
        """Reset scratchpad for a new investigation case."""
        self.clear()
        self.case_id = case_id
        self.target_company = target_company
        self.reporting_period = reporting_period

