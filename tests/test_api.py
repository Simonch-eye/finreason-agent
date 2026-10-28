"""
tests/test_api.py
=================
FinReason-Agent RESTful API 自动化集成测试套件

测试项：
1. /health 与 /api/v1/health 服务存活与元数据探针
2. /api/v1/sandbox/execute 隔离 DuckDB 沙箱确定性算术执行
3. /api/v1/parser/parse 财报文本/表格结构化提取
4. /api/v1/audit 端到端自主刑侦审计流水线调用
5. /api/v1/eval/scorecard 金标准双轨评测基准得分看板
6. /api/v1/export/datasets 轨迹微调数据集导出触发
"""

import pytest
from fastapi.testclient import TestClient

from finreason import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """测试服务健康探针"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "finreason-agent"

    resp_v1 = client.get("/api/v1/health")
    assert resp_v1.status_code == 200
    assert resp_v1.json()["status"] == "healthy"


def test_sandbox_execute_api(client):
    """测试沙箱执行 API"""
    payload = {
        "code": "result = (377.88 - 361.64) / 361.64\nprint(f'Done: {result:.4f}')"
    }
    resp = client.post("/api/v1/sandbox/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "Done: 0.0449" in data["stdout"]
    assert pytest.approx(data["result"], 0.001) == 0.0449


def test_parser_parse_api(client):
    """测试财报文本提取解析 API"""
    md_content = """# 康美药业 2018年年度报告
## 财务指标
| 会计科目 | 期末余额 | 期初余额 |
| :--- | :--- | :--- |
| 货币资金 [注1] | 377.88 | 361.64 |
| 短期借款 | 115.77 | 95.20 |

[注1] 货币资金中 299.44 亿元为定期存款及保函保证金受限。
"""
    resp = client.post("/api/v1/parser/parse", data={"text": md_content})
    assert resp.status_code == 200
    data = resp.json()
    assert data["company_name"] == "康美药业"
    assert data["reporting_period"] == "2018年年度"
    assert data["tables_count"] >= 1
    assert data["footnotes_count"] >= 1


def test_audit_api(client):
    """测试完整自主刑侦审计流水线 API"""
    report_text = """# 康美药业 2018年度报告
货币资金 377.88 亿元，短期借款 115.77 亿元，应付债券 168.00 亿元。
利息收入仅 2.8 亿元，利息支出高达 15.6 亿元。
[注1] 受限资金未在表内完整披露。
"""
    payload = {
        "query": "排查康美药业是否存在存贷双高与货币资金舞弊",
        "company_name": "康美药业",
        "reporting_period": "2018年度",
        "report_text": report_text,
    }
    resp = client.post("/api/v1/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["company_name"] == "康美药业"
    assert data["reporting_period"] == "2018年度"
    assert data["verdict"] in ["CRITICAL_FRAUD", "SUSPICIOUS_ANOMALY"]
    assert data["risk_score"] > 0.7
    assert len(data["evidence_chain"]) >= 1
    assert "Markdown" in data["memorandum_markdown"] or "# " in data["memorandum_markdown"]


def test_eval_scorecard_api(client):
    """测试双轨评测大屏得分 API (测试模式 20 案)"""
    resp = client.get("/api/v1/eval/scorecard?suite_size=20")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_cases"] == 20
    assert data["accuracy_pct"] == 100.0
    assert data["false_positives"] == 0
    assert data["false_negatives"] == 0


def test_export_datasets_api(client, tmp_path):
    """测试微调训练集导出 API"""
    export_dir = tmp_path / "test_api_exports"
    payload = {
        "output_dir": str(export_dir),
        "suite_size": 20,
    }
    resp = client.post("/api/v1/export/datasets", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["alpaca_records"] == 20
    assert (export_dir / "alpaca_sft.json").exists()
    assert (export_dir / "sharegpt_sft.jsonl").exists()
    assert (export_dir / "dpo_pairs.json").exists()
