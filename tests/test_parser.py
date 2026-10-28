"""Unit and integration tests for FinancialPDFParser, HierarchicalChunker, and FootnoteLinker.
"""

import io
import pandas as pd
import pytest

from finreason.parser import (
    ExtractedTable,
    FinancialPDFParser,
    FootnoteLink,
    FootnoteLinker,
    HierarchicalChunker,
)


def test_markdown_table_extraction():
    """Verify parsing and dataframe conversion of financial markdown tables."""
    parser = FinancialPDFParser()
    content = """
# 贵州茅台酒股份有限公司 2023年度报告
## 合并资产负债表

| 项目 | 期末余额 | 期初余额 | 附注 |
| :--- | :--- | :--- | :--- |
| 货币资金 | 58,487,975,419.12 | 58,274,321,123.00 | 附注七、1 |
| 应收账款 | 1,234,567.89 | 987,654.32 | 附注七、5 |
| 存货 | 46,000,000,000.00 | 38,000,000,000.00 | 附注七、8 |
"""
    report = parser.parse_markdown_or_text(content, source_name="maotai_2023")

    assert report.company_name == "贵州茅台酒股份有限公司"
    assert report.reporting_period == "2023年度"
    assert len(report.tables) == 1

    tbl = report.tables[0]
    df = tbl.to_dataframe()
    assert len(df) == 3
    assert "货币资金" in df["项目"].values
    assert tbl.footnote_refs.get("货币资金") == "附注七、1"
    assert tbl.footnote_refs.get("应收账款") == "附注七、5"


def test_continuation_table_merging():
    """Verify that multi-page tables across page boundaries are cleanly merged."""
    parser = FinancialPDFParser()

    t1 = ExtractedTable(
        title="合并资产负债表（流动资产）",
        page_start=1,
        page_end=1,
        headers=["项目", "金额"],
        rows=[["货币资金", "100"], ["交易性金融资产", "200"]],
    )
    t2 = ExtractedTable(
        title="续表",
        page_start=2,
        page_end=2,
        headers=["项目", "金额"],
        rows=[["应收账款", "300"], ["存货", "400"]],
    )

    merged = parser._merge_continuation_tables([t1, t2])
    assert len(merged) == 1
    assert merged[0].page_start == 1
    assert merged[0].page_end == 2
    assert len(merged[0].rows) == 4
    df = merged[0].to_dataframe()
    assert list(df["项目"]) == ["货币资金", "交易性金融资产", "应收账款", "存货"]


def test_hierarchical_chunking_breadcrumbs():
    """Verify that HierarchicalChunker builds full breadcrumb paths."""
    chunker = HierarchicalChunker()
    parser = FinancialPDFParser()

    report_text = """
第十节 财务报告
一、审计报告
标准无保留意见。

七、合并财务报表项目注释
1、货币资金
| 项目 | 期末余额 |
| :--- | :--- |
| 库存现金 | 1000.00 |
| 银行存款 | 50000000.00 |

5、应收账款
应收账款按账龄披露如下：
| 账龄 | 金额 |
| :--- | :--- |
| 1年以内 | 3000000.00 |
| 1至2年 | 500000.00 |
"""
    report = parser.parse_markdown_or_text(report_text)
    chunks = chunker.chunk_report(report)

    assert len(chunks) > 0

    # Look for the Accounts Receivable table chunk
    ar_table_chunks = [
        c for c in chunks if c.chunk_type == "table" and "1年以内" in c.content
    ]
    assert len(ar_table_chunks) == 1
    ar_chunk = ar_table_chunks[0]

    # Verify that breadcrumbs contain parent sections
    context_text = ar_chunk.get_contextual_text()
    assert "财务报告" in context_text or "应收账款" in context_text
    assert "1年以内" in context_text


def test_footnote_linking_bidirectional():
    """Verify bidirectional foreign key resolution between balance sheet accounts and footnotes."""
    parser = FinancialPDFParser()
    chunker = HierarchicalChunker()
    linker = FootnoteLinker()

    doc_text = """
第十节 财务报告
七、合并财务报表项目注释
主要会计报表如下：

| 项目 | 附注 | 期末金额 |
| :--- | :--- | :--- |
| 应收账款 | 附注七、5 | 3,500,000.00 |
| 在建工程 | 附注七、14 | 25,000,000.00 |

5、应收账款
应收账款前五名债务人明细披露：
| 单位名称 | 金额 | 占比 |
| :--- | :--- | :--- |
| 客户A | 2000000.00 | 57.1% |
| 客户B | 1000000.00 | 28.6% |

14、在建工程
重要在建工程项目变动情况：
| 工程名称 | 预算数 | 本期增加 |
| :--- | :--- | :--- |
| 医药生产基地二期 | 50000000.00 | 15000000.00 |
"""
    report = parser.parse_markdown_or_text(doc_text)
    chunks = chunker.chunk_report(report)
    links = linker.build_links(report, chunks)

    assert "应收账款" in links
    assert "在建工程" in links

    # Dereference Accounts Receivable footnote
    ar_link = linker.get_by_account("应收账款")
    assert ar_link is not None
    assert ar_link.footnote_ref == "附注七、5"
    assert len(ar_link.sub_tables) >= 1
    assert "客户A" in ar_link.sub_tables[0]["content"]

    # Dereference Construction-in-Progress footnote
    cip_link = linker.get_by_account("在建工程")
    assert cip_link is not None
    assert cip_link.footnote_ref == "附注七、14"
    assert len(cip_link.sub_tables) >= 1
    assert "医药生产基地二期" in cip_link.sub_tables[0]["content"]

    # Query by footnote tag
    ref_link = linker.get_by_ref("附注七、5")
    assert ref_link is not None
    assert ref_link.account_name == "应收账款"


def test_real_pdf_ingestion_pipeline(tmp_path):
    """Verify end-to-end PDF file parsing using pypdf."""
    from pypdf import PdfWriter

    pdf_path = tmp_path / "sample_report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    parser = FinancialPDFParser()
    report = parser.parse_pdf(str(pdf_path))

    assert report.total_pages == 1
    assert report.doc_id == "sample_report"
    assert 1 in report.pages_text

