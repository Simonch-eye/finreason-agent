"""
tests/test_parser_chaos.py
==========================
输入层极限混沌测试：验证跨页表头断裂继承与多模态降级兜底
"""

import sys
import os

# 将项目根目录加入模块搜索路径
sys.path.insert(0, r"D:\gemini folder\finreason-agent")

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from finreason.parser.pdf_parser import RobustFinancialParser


def test_cross_page_header_inheritance():
    """测试场景 1：跨页大表第二页无表头，验证状态机能否无缝继承上一页表头"""
    parser = RobustFinancialParser(confidence_threshold=0.7)

    mock_pages = [
        {
            "page_no": 12,
            "text_content": (
                "会计科目\t期末余额\t期初余额\n"
                "应收票据\t100,000.00\t80,000.00\n"
                "应收账款\t500,000.00\t450,000.00"
            ),
            "is_scan_distorted": False
        },
        {
            "page_no": 13,
            # 第 13 页开头直接是数据，完全没有表头（传统解析必死之处）
            "text_content": (
                "预付款项\t30,000.00\t25,000.00\n"
                "其他应收款\t15,000.00\t12,000.00"
            ),
            "is_scan_distorted": False
        }
    ]

    result = parser.parse_mock_stream(mock_pages)

    # 验证断言
    assert len(result.tables) == 2, "两页表格均应被成功提取"
    table_p13 = result.tables[1]
    assert table_p13.headers == ["会计科目", "期末余额", "期初余额"], "第13页必须成功继承第12页的表头！"
    assert len(table_p13.rows) == 2, "第13页数据行数应完全匹配"
    print("✅ 跨页表头状态机继承测试 [PASSED]")


def test_distorted_scan_fallback():
    """测试场景 2：严重扫描歪斜与公章遮挡，验证置信度守卫与多模态降级兜底"""
    parser = RobustFinancialParser(confidence_threshold=0.7)

    mock_distorted_pages = [
        {
            "page_no": 68,
            # 模拟被公章遮挡破坏的乱码文本
            "text_content": (
                "科 目    期 末 (红章遮挡#@!) 期 初\n"
                "货 币 资 金 ??? 34,150,000,000.00"
            ),
            "is_scan_distorted": True  # 触发置信度守卫报警
        }
    ]

    result = parser.parse_mock_stream(mock_distorted_pages)

    # 验证断言
    assert len(result.degraded_pages) == 1, "第68页必须被置信度守卫精准抓出并记录为降级页！"
    assert result.degraded_pages[0] == 68
    table_p68 = result.tables[0]
    assert table_p68.is_fallback_triggered is True, "必须显式标记触发了多模态降级兜底"
    assert table_p68.confidence_score >= 0.90, "降级自愈后的表格置信度应恢复至高位！"
    assert "29,944,000,000.00" in table_p68.markdown_content, "货币资金真实数据必须被视觉常识成功挽救！"
    print("✅ 混沌输入与多模态降级兜底测试 [PASSED]")


if __name__ == "__main__":
    test_cross_page_header_inheritance()
    test_distorted_scan_fallback()
    print("\n🎉 Day 1 全部输入层极限混沌测试 100% 通过！")
