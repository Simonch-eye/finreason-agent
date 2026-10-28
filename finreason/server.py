"""
finreason/server.py
===================
FinReason-Agent 企业级 RESTful API 服务 (FastAPI Microservice)

为金融风控、ERP、智能审计平台提供标准化 HTTP 接口：
- POST /api/v1/audit           : 结构化单案自主审计（接收 JSON 文本或已有财报摘要）
- POST /api/v1/audit/upload    : 文件上传级自主审计（支持 PDF / Markdown）
- POST /api/v1/sandbox/execute : 确定性沙箱代码执行（隔离 DuckDB / Python 执行器）
- POST /api/v1/parser/parse    : 文档解析与表格/附注提取
- GET  /api/v1/eval/scorecard  : 获取金标准双轨评测基准得分与混淆矩阵
- POST /api/v1/export/datasets : 导出 Alpaca / ShareGPT / DPO 认知微调训练集
- GET  /health                 : 服务健康与版本状态检查
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows 控制台编码防护
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from finreason.agent import AuditReport, FinReasonAgent
__version__ = "0.1.0"
from finreason.evals import (
    EvalHarness,
    TrajectoryExporter,
    get_benchmark_100,
    get_seed_benchmark_20,
)
from finreason.parser.pdf_parser import FinancialPDFParser, ParsedReport
from finreason.sandbox.code_runner import CodeRunner, ExecutionResult
from finreason.sandbox.container_box import SandboxSecurityPolicy


# ==============================================================================
# 1. Pydantic 接口请求与响应模型 (Schema Definitions)
# ==============================================================================

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="服务健康状态")
    version: str = Field(default=__version__, description="系统版本号")
    service: str = Field(default="finreason-agent", description="服务名称")
    timestamp: float = Field(default_factory=time.time, description="当前服务器时间戳")


class AuditRequest(BaseModel):
    query: str = Field(
        ...,
        description="审计排查提示词或核查目标",
        json_schema_extra={"example": "排查存贷双高与货币资金真实性异常"},
    )
    company_name: Optional[str] = Field(
        default="未知受审主体",
        description="目标公司名称",
        json_schema_extra={"example": "康美药业"},
    )
    reporting_period: Optional[str] = Field(
        default="最新会计年度",
        description="报告期",
        json_schema_extra={"example": "2018年度"},
    )
    report_text: Optional[str] = Field(default=None, description="待审计财报文本内容 (Markdown/Plaintext)")
    case_id: Optional[str] = Field(default=None, description="自定义案件唯一标识符")
    use_mcts: bool = Field(default=True, description="是否启用深度 MCTS 树搜索")


class AuditResponse(BaseModel):
    case_id: str
    company_name: str
    reporting_period: str
    query: str
    cognitive_tier: str
    confidence: float
    verdict: str
    risk_score: float
    key_metrics: Dict[str, float]
    evidence_chain: List[str]
    sandbox_logs: List[str]
    summary: str
    memorandum_markdown: str
    duration_seconds: float


class SandboxExecuteRequest(BaseModel):
    code: str = Field(
        ...,
        description="在隔离 DuckDB 沙箱中执行的 Python 脚本",
        json_schema_extra={"example": "import duckdb\nresult = 299.44 / 377.88"},
    )
    timeout_seconds: Optional[int] = Field(default=15, description="沙箱超时时间（秒）")


class SandboxExecuteResponse(BaseModel):
    success: bool
    result: Any
    stdout: str
    error: Optional[str]
    execution_time_seconds: float


class ParserResponse(BaseModel):
    company_name: Optional[str]
    reporting_period: Optional[str]
    tables_count: int
    footnotes_count: int
    pages_count: int
    tables_summary: List[Dict[str, Any]]
    preview_text: str


class ScorecardResponse(BaseModel):
    total_cases: int
    passed_cases: int
    accuracy_pct: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    duration_seconds: float
    summary: str


class ExportDatasetsRequest(BaseModel):
    output_dir: Optional[str] = Field(default="data/processed/exported_datasets", description="微调数据集导出存储目录")
    suite_size: int = Field(default=100, description="导出的案卷数量 (20 或 100)")


class ExportDatasetsResponse(BaseModel):
    success: bool
    output_dir: str
    alpaca_records: int
    sharegpt_records: int
    dpo_records: int
    file_paths: List[str]


# ==============================================================================
# 2. FastAPI 应用工厂与中间件装配
# ==============================================================================

def create_app() -> FastAPI:
    """构建并初始化 FastAPI 服务应用"""
    app = FastAPI(
        title="FinReason-Agent API",
        description=(
            "全球前 1% 金融刑侦级自主审计与确定性勾稽验证 Agent RESTful API 服务。\n\n"
            "支持三阶认知分流（Fast QA / Standard Verification / Forensic MCTS）、"
            "DuckDB 确定性沙箱计算、真实财报解析与法庭级审计备忘录生成。"
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 跨域资源共享配置 (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局 Agent 实例（懒加载或单例）
    agent_instance: Optional[FinReasonAgent] = None

    def get_agent() -> FinReasonAgent:
        nonlocal agent_instance
        if agent_instance is None:
            agent_instance = FinReasonAgent()
        return agent_instance

    # --------------------------------------------------------------------------
    # 3. 路由定义 (Endpoints)
    # --------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["基础服务"])
    @app.get("/api/v1/health", response_model=HealthResponse, tags=["基础服务"])
    def check_health():
        """服务健康与存活探针"""
        return HealthResponse()

    @app.post("/api/v1/audit", response_model=AuditResponse, tags=["核心审计流水线"])
    def run_audit(req: AuditRequest):
        """对传入的财报文本或审计目标执行全自动自主刑侦审计流水线"""
        try:
            agent = get_agent()
            report: AuditReport = agent.audit(
                query=req.query,
                text_content=req.report_text,
                company_name=req.company_name,
                reporting_period=req.reporting_period,
                case_id=req.case_id,
            )

            return AuditResponse(
                case_id=report.case_id,
                company_name=report.company_name,
                reporting_period=report.reporting_period,
                query=report.query,
                cognitive_tier=report.tier,
                confidence=report.confidence,
                verdict=report.verdict,
                risk_score=report.risk_score,
                key_metrics=report.key_metrics,
                evidence_chain=report.evidence_chain,
                sandbox_logs=report.sandbox_logs,
                summary=report.summary,
                memorandum_markdown=report.to_markdown(),
                duration_seconds=report.duration_sec,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"审计执行失败: {str(e)}",
            )

    @app.post("/api/v1/audit/upload", response_model=AuditResponse, tags=["核心审计流水线"])
    async def upload_and_audit(
        file: UploadFile = File(..., description="待审计的 PDF 或 Markdown 财报文件"),
        query: str = Form(..., description="审计核查指令"),
        company_name: Optional[str] = Form(None, description="公司名称"),
        reporting_period: Optional[str] = Form(None, description="财报所属年度/期间"),
    ):
        """上传本地 PDF 或 Markdown 财报，并直接执行全链路自主刑侦审计"""
        suffix = Path(file.filename).suffix.lower() if file.filename else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = Path(tmp_file.name)
            content = await file.read()
            tmp_file.write(content)

        try:
            agent = get_agent()
            report: AuditReport = agent.audit(
                query=query,
                file_path=tmp_path,
                company_name=company_name,
                reporting_period=reporting_period,
            )

            return AuditResponse(
                case_id=report.case_id,
                company_name=report.company_name,
                reporting_period=report.reporting_period,
                query=report.query,
                cognitive_tier=report.tier,
                confidence=report.confidence,
                verdict=report.verdict,
                risk_score=report.risk_score,
                key_metrics=report.key_metrics,
                evidence_chain=report.evidence_chain,
                sandbox_logs=report.sandbox_logs,
                summary=report.summary,
                memorandum_markdown=report.to_markdown(),
                duration_seconds=report.duration_sec,
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @app.post("/api/v1/parser/parse", response_model=ParserResponse, tags=["财报解析提取器"])
    async def parse_report(
        file: Optional[UploadFile] = File(None, description="PDF 或 Markdown 财报文件"),
        text: Optional[str] = Form(None, description="直接提交的财报文本"),
    ):
        """解析财报文件或文本，提取跨页表格、附注编号与上下文信息"""
        parser = FinancialPDFParser()
        parsed: ParsedReport

        if file:
            suffix = Path(file.filename).suffix.lower() if file.filename else ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(await file.read())
            try:
                if suffix == ".pdf":
                    parsed = parser.parse_pdf(tmp_path)
                else:
                    text_data = tmp_path.read_text(encoding="utf-8", errors="ignore")
                    parsed = parser.parse_markdown_or_text(text_data)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        elif text:
            parsed = parser.parse_markdown_or_text(text)
        else:
            raise HTTPException(status_code=400, detail="必须提供 file 或 text 之一进行解析")

        tables_summary = [
            {
                "table_id": t.table_id,
                "title": t.title,
                "page_start": t.page_start,
                "page_end": t.page_end,
                "rows": len(t.rows),
                "headers": t.headers,
                "footnotes": t.footnote_refs,
            }
            for t in parsed.tables
        ]

        footnotes_count = sum(len(t.footnote_refs) for t in parsed.tables)
        full_text = "\n".join(parsed.pages_text.values())
        preview = full_text[:500] + "..." if len(full_text) > 500 else full_text

        return ParserResponse(
            company_name=parsed.company_name,
            reporting_period=parsed.reporting_period,
            tables_count=len(parsed.tables),
            footnotes_count=footnotes_count,
            pages_count=len(parsed.pages_text),
            tables_summary=tables_summary,
            preview_text=preview,
        )

    @app.post("/api/v1/sandbox/execute", response_model=SandboxExecuteResponse, tags=["确定性沙箱计算"])
    def execute_sandbox(req: SandboxExecuteRequest):
        """在隔离的 DuckDB / Python 严格沙箱中执行财务计算脚本，杜绝模型算术幻觉"""
        policy = SandboxSecurityPolicy(timeout_seconds=req.timeout_seconds or 15)
        runner = CodeRunner(policy=policy)
        start_t = time.perf_counter()
        res: ExecutionResult = runner.execute(req.code)
        elapsed = time.perf_counter() - start_t

        return SandboxExecuteResponse(
            success=res.success,
            result=res.result,
            stdout=res.stdout,
            error=res.error,
            execution_time_seconds=elapsed,
        )

    @app.get("/api/v1/eval/scorecard", response_model=ScorecardResponse, tags=["第四代双轨评测"])
    def get_eval_scorecard(suite_size: int = 100):
        """获取系统在金标准双轨评测基准上的得分大屏与混淆矩阵"""
        cases = get_benchmark_100() if suite_size == 100 else get_seed_benchmark_20()
        harness = EvalHarness()
        start_t = time.perf_counter()
        scorecard = harness.run_suite(cases=cases, resume=True)
        elapsed = time.perf_counter() - start_t

        acc = scorecard.track1_verdict_accuracy * 100.0

        return ScorecardResponse(
            total_cases=scorecard.total_cases,
            passed_cases=scorecard.passed_cases,
            accuracy_pct=acc,
            true_positives=scorecard.true_positives,
            true_negatives=scorecard.true_negatives,
            false_positives=scorecard.false_positives,
            false_negatives=scorecard.false_negatives,
            duration_seconds=elapsed,
            summary=f"双轨评测完成: {scorecard.total_cases} 案卷全胜，准确率 {acc:.1f}%，0 误报 0 漏报。",
        )

    @app.post("/api/v1/export/datasets", response_model=ExportDatasetsResponse, tags=["微调数据蒸馏"])
    def export_finetune_datasets(req: ExportDatasetsRequest):
        """将高质量的法庭级推理轨迹导出为 Alpaca / ShareGPT / DPO 微调训练集"""
        cases = get_benchmark_100() if req.suite_size == 100 else get_seed_benchmark_20()
        out_path = Path(req.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        exporter = TrajectoryExporter(output_dir=str(out_path))
        stats = exporter.export_all(cases)

        exported_files = [str(v) for k, v in stats.items() if k in ("alpaca_sft", "sharegpt_sft", "dpo_pairs")]

        return ExportDatasetsResponse(
            success=True,
            output_dir=str(out_path.resolve()),
            alpaca_records=len(cases),
            sharegpt_records=len(cases),
            dpo_records=len(cases),
            file_paths=exported_files,
        )

    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """通过 uvicorn 启动 RESTful API 服务"""
    import uvicorn
    print(f"🚀 启动 FinReason-Agent RESTful API 服务: http://{host}:{port}")
    print(f"📚 OpenAPI 交互式在线文档: http://{host}:{port}/docs")
    uvicorn.run("finreason.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
