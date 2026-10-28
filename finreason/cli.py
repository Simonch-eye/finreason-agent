"""
finreason/cli.py
================
FinReason-Agent 工业级终端命令行交互工具 (Command Line Interface)

支持指令：
  finreason audit <file> --query "..."   # 对单份财报执行全自动自主刑侦审计
  finreason eval --suite [100|20]        # 运行第四代双轨评测大屏 (Dual-Track Benchmark)
  finreason export-data --output-dir ... # 导出 100 案 SFT / ShareGPT / DPO 微调数据集
  finreason demo                         # 运行交互式全景实战演示
  finreason version                      # 查看当前版本与架构信息
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台编码防护
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from finreason import __version__, AuditReport, FinReasonAgent
from finreason.evals import (
    EvalHarness,
    TrajectoryExporter,
    get_benchmark_100,
    get_seed_benchmark_20,
)


def cmd_audit(args: argparse.Namespace) -> int:
    """执行财报自主刑侦审计"""
    print("=" * 70)
    print("       FinReason-Agent 金融刑侦级自主审计系统 (Audit Mode)       ")
    print("=" * 70)

    file_path = args.file
    query = args.query

    if file_path and not Path(file_path).exists():
        print(f"❌ 错误: 指定的财报文件不存在: {file_path}")
        return 1

    print(f"📋 审计目标: {query}")
    if file_path:
        print(f"📄 载入财报: {file_path}")
    print("🚀 启动主脑认知路由与沙箱分析...")

    agent = FinReasonAgent()
    report: AuditReport = agent.audit(
        query=query,
        file_path=file_path,
        company_name=args.company,
        reporting_period=args.period,
    )

    print("\n" + report.to_markdown())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        print(f"\n💾 审计意见书已成功保存至: {out_path.resolve()}")

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """运行第四代双轨评测基准"""
    suite_num = args.suite
    cases = get_benchmark_100() if suite_num == 100 else get_seed_benchmark_20()
    ckpt_file = args.checkpoint or (
        "data/processed/eval_100_ckpt.json" if suite_num == 100 else "data/processed/eval_test_ckpt.json"
    )

    print("=" * 70)
    print(f"    FinReason-Agent 第四代双轨评测考场 (Suite: {suite_num} 案)    ")
    print("=" * 70)
    print(f"• 考题总数: {len(cases)} 案")
    print(f"• 断点缓存: {ckpt_file} (断点续跑: {args.resume})")
    print("• 正在调取双轨直判与双盲去偏打分...")

    harness = EvalHarness(checkpoint_file=ckpt_file)
    summary = harness.run_suite(cases=cases, resume=args.resume)
    print("\n" + summary.to_console_dashboard())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """导出用于开源大模型微调的黄金轨迹 (SFT / ShareGPT / DPO)"""
    out_dir = args.output_dir
    print("=" * 70)
    print("   FinReason-Agent 刑侦审计黄金微调轨迹导出器 (Trajectory Exporter)   ")
    print("=" * 70)
    print(f"• 目标目录: {out_dir}")
    print("• 正在由 100 案构建 Alpaca、ShareGPT 与 DPO 偏好对...")

    cases = get_benchmark_100()
    exporter = TrajectoryExporter(output_dir=out_dir)
    paths = exporter.export_all(cases)

    print("\n✅ 数据集导出完成:")
    for fmt_name, f_path in paths.items():
        if fmt_name != "total_samples":
            print(f"  • [{fmt_name.upper()}] -> {f_path}")


    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """运行交互式演示流程"""
    from run_demo import main as run_demo_main
    run_demo_main()
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """启动 Streamlit 可视化刑侦审计工作台"""
    import subprocess
    app_path = Path(__file__).resolve().parent.parent / "web_app.py"
    port = str(args.port or 8501)
    print(f"🚀 正在启动 FinReason-Agent 可视化工作台 (Port: {port})...")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", port])


def cmd_serve(args: argparse.Namespace) -> int:
    """启动 FastAPI RESTful 生产级后端微服务"""
    from finreason.server import run_server
    run_server(host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """打印版本与系统状态"""
    print(f"FinReason-Agent: v{__version__}")
    print("Architecture: Autonomous Financial Intelligence & Forensic Audit Agent")
    print("Core Capabilities: Zero Arithmetic Hallucination (DuckDB), Cognitive Tiering, ToT/MCTS, Dual-Track Evals")
    return 0


def main() -> int:

    """CLI 主入口函数"""
    parser = argparse.ArgumentParser(
        prog="finreason",
        description="FinReason-Agent: 全球前 1% 金融刑侦级自主审计与勾稽验证系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"finreason-agent v{__version__}")

    subparsers = parser.add_subparsers(dest="subcommand", help="可用子命令")

    # 1. audit 命令
    p_audit = subparsers.add_parser("audit", help="对输入财报或提问执行自主刑侦审计")
    p_audit.add_argument("file", nargs="?", default=None, help="待审财报文件路径 (.pdf / .md / .txt)")
    p_audit.add_argument("-q", "--query", required=True, help="审计任务目标或舞弊排查疑点")
    p_audit.add_argument("-c", "--company", default=None, help="受审公司名称 (可选)")
    p_audit.add_argument("-p", "--period", default=None, help="审计报告期 (可选)")
    p_audit.add_argument("-o", "--output", default=None, help="输出 Markdown 审计意见书文件路径")

    # 2. eval 命令
    p_eval = subparsers.add_parser("eval", help="运行第四代双轨评测考场大屏")
    p_eval.add_argument("--suite", type=int, choices=[20, 100], default=100, help="评测集规模: 100 或 20 (默认: 100)")
    p_eval.add_argument("--checkpoint", default=None, help="自定义断点续跑文件路径")
    p_eval.add_argument("--no-resume", dest="resume", action="store_false", default=True, help="不使用断点缓存，强制全量重跑")

    # 3. export-data 命令
    p_export = subparsers.add_parser("export-data", help="导出用于大模型微调的 Alpaca / ShareGPT / DPO 数据集")
    p_export.add_argument("-o", "--output-dir", default="data/processed/finetune_trajectories", help="导出目录路径")

    # 4. demo 命令
    subparsers.add_parser("demo", help="运行系统核心能力综合演示")

    # 5. version 命令
    subparsers.add_parser("version", help="查看系统版本与核心架构")

    # 6. ui 命令
    p_ui = subparsers.add_parser("ui", help="启动 Streamlit 交互式金融刑侦审计工作台")
    p_ui.add_argument("--port", type=int, default=8501, help="Web 服务器端口 (默认: 8501)")

    # 7. serve 命令
    p_serve = subparsers.add_parser("serve", help="启动 FastAPI RESTful 后端微服务")
    p_serve.add_argument("--host", default="0.0.0.0", help="监听主机地址 (默认: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="服务端口 (默认: 8000)")
    p_serve.add_argument("--reload", action="store_true", default=False, help="启用热重载 (开发模式)")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return 0

    dispatch = {
        "audit": cmd_audit,
        "eval": cmd_eval,
        "export-data": cmd_export,
        "demo": cmd_demo,
        "version": cmd_version,
        "ui": cmd_ui,
        "serve": cmd_serve,
    }

    handler = dispatch.get(args.subcommand)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
