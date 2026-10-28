"""FinReason-Agent: Financial Intelligence & Forensic Audit Agent with Deterministic Verification.
"""

from finreason.agent import AuditReport, FinReasonAgent
from finreason.core import CognitiveRouter, CognitiveTier, MCTSEngine, ToTEngine
from finreason.parser import (
    DocumentChaosGuard,
    ExtractedTable,
    FinancialPDFParser,
    FootnoteLinker,
    HierarchicalChunker,
    ParseResult,
    ParsedReport,
    RobustFinancialParser,
    TableBlock,
)
from finreason.sandbox import CodeRunner
from finreason.server import app, run_server

__version__ = "0.1.0"

__all__ = [
    "FinReasonAgent",
    "AuditReport",
    "CognitiveRouter",
    "CognitiveTier",
    "MCTSEngine",
    "ToTEngine",
    "CodeRunner",
    "FinancialPDFParser",
    "RobustFinancialParser",
    "ExtractedTable",
    "TableBlock",
    "ParsedReport",
    "ParseResult",
    "DocumentChaosGuard",
    "HierarchicalChunker",
    "FootnoteLinker",
    "app",
    "run_server",
]
