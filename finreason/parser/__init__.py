"""Financial report parsing, layout extraction, hierarchical chunking, and footnote linking.
"""

from finreason.parser.chunker import HierarchicalChunk, HierarchicalChunker
from finreason.parser.footnote_linker import FootnoteLink, FootnoteLinker
from finreason.parser.pdf_parser import (
    DocumentChaosGuard,
    ExtractedTable,
    FinancialPDFParser,
    ParseResult,
    ParsedReport,
    RobustFinancialParser,
    TableBlock,
)

__all__ = [
    "FinancialPDFParser",
    "RobustFinancialParser",
    "ExtractedTable",
    "TableBlock",
    "ParsedReport",
    "ParseResult",
    "DocumentChaosGuard",
    "HierarchicalChunker",
    "HierarchicalChunk",
    "FootnoteLinker",
    "FootnoteLink",
]
