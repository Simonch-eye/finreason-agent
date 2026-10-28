"""Hierarchical Chunking (父子层级分块) for financial reports.
Preserves structural breadcrumbs (Section -> Subsection -> Account -> Table/Text) to eliminate lost context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from finreason.parser.pdf_parser import ExtractedTable, ParsedReport


class HierarchicalChunk(BaseModel):
    """A semantic chunk preserving hierarchical lineage and breadcrumbs."""
    id: str = Field(description="Unique chunk identifier")
    level: int = Field(description="Hierarchy depth level (1: Section, 2: Subsection, 3: Account Item, 4: Table/Body)")
    title: str = Field(description="Header or component title")
    breadcrumbs: List[str] = Field(default_factory=list, description="Full lineage path (e.g. [财务报告, 合并项目注释, 应收账款])")
    content: str = Field(description="Chunk payload text or markdown table")
    page_num: int = Field(default=1, description="Source page number")
    parent_id: Optional[str] = Field(default=None, description="Parent chunk id")
    children_ids: List[str] = Field(default_factory=list, description="Child chunk ids")
    chunk_type: str = Field(default="text", description="Type: section_header, table, text, footnote")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata (e.g. account_name, amount)")

    def get_contextual_text(self) -> str:
        """Render chunk with full breadcrumb header for high-accuracy retrieval."""
        path_str = " > ".join(self.breadcrumbs) if self.breadcrumbs else self.title
        return f"[路径: {path_str} | 页码: P{self.page_num}]\n{self.content}"


class HierarchicalChunker:
    """Parses flat financial report text and tables into a structured tree of hierarchical chunks."""

    # Heading patterns in Chinese financial reports
    L1_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百]+节|SECTION\s+\d+)\s*([^\n\r]+)", re.IGNORECASE)
    L2_PATTERN = re.compile(r"^([一二三四五六七八九十]+、)\s*([^\n\r]+)")
    L3_PATTERN = re.compile(r"^(?:（[一二三四五六七八九十]+）|(\d+、))\s*([^\n\r]+)")
    L4_PATTERN = re.compile(r"^(?:\(\d+\)|(\d+\.\d+))\s*([^\n\r]+)")

    def __init__(self, max_chunk_chars: int = 2000):
        self.max_chunk_chars = max_chunk_chars

    def chunk_report(self, report: ParsedReport) -> List[HierarchicalChunk]:
        """Convert a parsed report into a flat list of interrelated hierarchical chunks."""
        chunks: List[HierarchicalChunk] = []

        # Track active hierarchy stack: [(level, chunk_id, title)]
        stack: List[tuple[int, str, str]] = []

        chunk_counter = 0

        def current_breadcrumbs() -> List[str]:
            return [s[2] for s in stack]

        def current_parent_id() -> Optional[str]:
            return stack[-1][1] if stack else None

        for page_num in sorted(report.pages_text.keys()):
            page_text = report.pages_text[page_num]
            lines = page_text.splitlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                # Check headings
                heading_level, heading_title = self._detect_heading(line)
                if heading_level is not None:
                    # Pop stack until finding a higher-level ancestor
                    while stack and stack[-1][0] >= heading_level:
                        stack.pop()

                    chunk_counter += 1
                    cid = f"chunk_{page_num}_{chunk_counter}"
                    parent_id = current_parent_id()

                    chunk = HierarchicalChunk(
                        id=cid,
                        level=heading_level,
                        title=heading_title,
                        breadcrumbs=current_breadcrumbs() + [heading_title],
                        content=line,
                        page_num=page_num,
                        parent_id=parent_id,
                        chunk_type="section_header",
                        metadata={"is_heading": True},
                    )
                    chunks.append(chunk)
                    stack.append((heading_level, cid, heading_title))
                    i += 1
                    continue

                # Check if this line is part of a Markdown table or paragraph block
                if line.startswith("|") and line.endswith("|"):
                    # Collect full table
                    table_lines = [line]
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                        table_lines.append(lines[i].strip())
                        i += 1
                    table_content = "\n".join(table_lines)

                    chunk_counter += 1
                    cid = f"chunk_{page_num}_{chunk_counter}"
                    tbl_chunk = HierarchicalChunk(
                        id=cid,
                        level=len(stack) + 1,
                        title=stack[-1][2] if stack else f"Table_P{page_num}",
                        breadcrumbs=current_breadcrumbs() + [f"Table_P{page_num}"],
                        content=table_content,
                        page_num=page_num,
                        parent_id=current_parent_id(),
                        chunk_type="table",
                        metadata={"format": "markdown_table"},
                    )
                    chunks.append(tbl_chunk)
                    continue

                # Normal paragraph text: accumulate up to max_chunk_chars or until next heading
                para_lines = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line:
                        i += 1
                        continue
                    # Break if heading or table starts
                    if self._detect_heading(next_line)[0] is not None or (next_line.startswith("|") and next_line.endswith("|")):
                        break
                    para_lines.append(next_line)
                    if sum(len(l) for l in para_lines) >= self.max_chunk_chars:
                        i += 1
                        break
                    i += 1

                chunk_counter += 1
                cid = f"chunk_{page_num}_{chunk_counter}"
                para_chunk = HierarchicalChunk(
                    id=cid,
                    level=len(stack) + 1,
                    title=stack[-1][2] if stack else "General",
                    breadcrumbs=current_breadcrumbs(),
                    content="\n".join(para_lines),
                    page_num=page_num,
                    parent_id=current_parent_id(),
                    chunk_type="text",
                )
                chunks.append(para_chunk)

        # Build children_ids relationships
        id_to_chunk = {c.id: c for c in chunks}
        for c in chunks:
            if c.parent_id and c.parent_id in id_to_chunk:
                id_to_chunk[c.parent_id].children_ids.append(c.id)

        return chunks

    def _detect_heading(self, line: str) -> tuple[Optional[int], str]:
        """Detect hierarchical heading level and clean title."""
        if len(line) > 80:  # Headings are generally short lines
            return None, ""

        m1 = self.L1_PATTERN.match(line)
        if m1:
            return 1, line

        m2 = self.L2_PATTERN.match(line)
        if m2:
            return 2, line

        m3 = self.L3_PATTERN.match(line)
        if m3:
            return 3, line

        m4 = self.L4_PATTERN.match(line)
        if m4:
            return 4, line

        return None, ""
