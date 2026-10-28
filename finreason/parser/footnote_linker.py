"""Bidirectional anchor between main financial statements and in-depth footnote tables.
Enables forensic cross-verification and instant drill-down from primary accounts to sub-ledgers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from finreason.parser.chunker import HierarchicalChunk
from finreason.parser.pdf_parser import ExtractedTable, ParsedReport


class FootnoteLink(BaseModel):
    """Foreign key link connecting a main financial account to its deep footnote details."""
    account_name: str = Field(description="Primary account name in the main financial statement (e.g. 应收账款, 在建工程)")
    footnote_ref: str = Field(description="Footnote reference tag in the main table (e.g. 附注七、5 or 注释14)")
    normalized_id: str = Field(description="Normalized key, e.g. 7_5 or note_14")
    footnote_title: str = Field(default="", description="Title of the corresponding footnote section")
    page_num: int = Field(default=1, description="Page number where the footnote appears")
    sub_tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted detailed tables within the footnote")
    footnote_text: str = Field(default="", description="Accompanying disclosure text and accounting policies")

    def to_summary(self) -> str:
        """Render a concise summary of the footnote link and its sub-tables."""
        sub_tbl_count = len(self.sub_tables)
        return (
            f"[附注穿透: {self.account_name} -> {self.footnote_ref} ({self.footnote_title}) | P{self.page_num}]\n"
            f"含 {sub_tbl_count} 个明细表格与附注披露文本: {self.footnote_text[:200]}..."
        )


class FootnoteLinker:
    """Builds and queries the bidirectional index between main financial tables and footnote disclosures."""

    def __init__(self):
        self.account_to_link: Dict[str, FootnoteLink] = {}
        self.ref_to_link: Dict[str, FootnoteLink] = {}

    def normalize_ref(self, ref_str: str) -> str:
        """Normalize various footnote reference strings (附注七、5, 注释5, 5, (七)5) into a uniform id."""
        if not ref_str:
            return ""

        # Map Chinese numerals to digits
        cn_num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        clean = ref_str.strip().replace("附注", "").replace("注释", "").replace("注", "").strip()

        # Handle combinations like "七、5" or "七.5"
        m = re.match(r"([一二三四五六七八九十]+)[\.\、](\d+)", clean)
        if m:
            sec_num = cn_num.get(m.group(1), m.group(1))
            item_num = m.group(2)
            return f"ref_{sec_num}_{item_num}"

        # Handle simple digits or single numbers
        digit_match = re.search(r"\d+", clean)
        if digit_match:
            return f"ref_{digit_match.group(0)}"

        return f"ref_{clean}"

    def build_links(
        self,
        report: ParsedReport,
        chunks: Optional[List[HierarchicalChunk]] = None,
    ) -> Dict[str, FootnoteLink]:
        """Scan report tables and text to construct foreign key links."""
        self.account_to_link.clear()
        self.ref_to_link.clear()

        # 1. Collect all footnote references from extracted tables
        account_refs: Dict[str, str] = {}
        for tbl in report.tables:
            if tbl.footnote_refs:
                account_refs.update(tbl.footnote_refs)

        # 2. Build index of footnote section headers and content
        footnote_sections: Dict[str, Dict[str, Any]] = {}

        if chunks:
            for c in chunks:
                # Detect note sections e.g. "5、应收账款" or "14、在建工程"
                m = re.match(r"^(?:（[一二三四五六七八九十]+）|(\d+))[\、\.]\s*([\u4e00-\u9fa5A-Za-z0-9_]+)", c.title)
                if m:
                    item_num = m.group(1) or m.group(0)
                    account_hint = m.group(2) if len(m.groups()) >= 2 else ""
                    norm_key = self.normalize_ref(item_num)

                    if norm_key not in footnote_sections:
                        footnote_sections[norm_key] = {
                            "title": c.title,
                            "page_num": c.page_num,
                            "account_hint": account_hint,
                            "text_parts": [],
                            "sub_tables": [],
                        }

                    if c.chunk_type == "table":
                        footnote_sections[norm_key]["sub_tables"].append({"content": c.content, "page": c.page_num})
                    else:
                        footnote_sections[norm_key]["text_parts"].append(c.content)

        # 3. Correlate main statement accounts with indexed footnote sections
        for account, raw_ref in account_refs.items():
            norm_id = self.normalize_ref(raw_ref)
            target_sec = footnote_sections.get(norm_id)

            # Fallback fuzzy match by account name if ref match not found directly
            if not target_sec:
                for k, sec in footnote_sections.items():
                    if sec["account_hint"] and (sec["account_hint"] in account or account in sec["account_hint"]):
                        target_sec = sec
                        norm_id = k
                        break

            link = FootnoteLink(
                account_name=account,
                footnote_ref=raw_ref,
                normalized_id=norm_id,
                footnote_title=target_sec["title"] if target_sec else f"Footnote_{raw_ref}",
                page_num=target_sec["page_num"] if target_sec else 1,
                sub_tables=target_sec["sub_tables"] if target_sec else [],
                footnote_text="\n".join(target_sec["text_parts"]) if target_sec else "",
            )

            self.account_to_link[account] = link
            self.ref_to_link[raw_ref] = link
            self.ref_to_link[norm_id] = link

        return self.account_to_link

    def get_by_account(self, account_name: str) -> Optional[FootnoteLink]:
        """Look up footnote link by account name (e.g. 应收账款)."""
        if account_name in self.account_to_link:
            return self.account_to_link[account_name]
        # Partial match
        for acc, link in self.account_to_link.items():
            if acc in account_name or account_name in acc:
                return link
        return None

    def get_by_ref(self, ref_tag: str) -> Optional[FootnoteLink]:
        """Look up footnote link by raw reference or normalized id."""
        if ref_tag in self.ref_to_link:
            return self.ref_to_link[ref_tag]
        norm = self.normalize_ref(ref_tag)
        return self.ref_to_link.get(norm)
