# FinReason-Agent 项目主宪章与系统架构蓝图 (Master Specification)

> **目标**：打造全球前 1% 的金融刑侦级自主审计与勾稽验证 Agent 系统（Autonomous Financial Intelligence & Audit System）。
> **核心原则**：
> 1. **零算术幻觉（Zero Arithmetic Hallucination）**：所有数字加减乘除、比率拆解、勾稽计算，100% 走 Python/DuckDB 隔离沙箱。
> 2. **自适应成本分流（Cognitive Tiering）**：简单查账走快速通道（Tier 1）；公式化指标走确定性状态机（Tier 2）；复杂舞弊排查走 MCTS/ToT 刑侦树与自反思（Tier 3）。
> 3. **四层分立记忆（Hierarchical Memory）**：草稿放工作记忆、专案日志放情节记忆、行业通识放语义记忆、验证代码放程序记忆。
> 4. **客观可验证评测（Dual-Track Verifiable Evals）**：以 100 套真实上市公司年报为基准，硬核代码直判 + 协议双盲去偏裁判。

---

## 一、系统架构全景 (System Architecture)

```
                            [ 用户输入 / 待审年报 PDF ]
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ 1. 数据对齐层 (Parser & Linking)                       │
            │    • 视觉多模态版面还原 (Markdown/HTML Tables)          │
            │    • 父子层级分块 (Hierarchical Chunking)               │
            │    • 正文主科目与深层附注外键双向绑定 (Footnote Links)   │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ 2. 门房认知路由器 (Cognitive Router)                   │
            │    • 基于语法受限解码 (FSM/Instructor) 判定复杂度等级  │
            └──────┬─────────────────────┼─────────────────────┬─────┘
                   │                     │                     │
                   ▼ (Tier 1: 直觉级)     ▼ (Tier 2: 规范级)    ▼ (Tier 3: 刑侦级)
            ┌──────────────┐      ┌──────────────┐      ┌────────────────────────┐
            │ Fast-Path    │      │ 确定性状态机 │      │ 深度推理刑侦树         │
            │ (混合检索QA) │      │ (单流沙箱验算)│     │ (ToT/MCTS + Reflexion) │
            └──────────────┘      └──────┬───────┘      └───────────┬────────────┘
                                         │                          │
                                         └─────────────┬────────────┘
                                                       │ 调度确定性工具
                                                       ▼
            ┌────────────────────────────────────────────────────────┐
            │ 3. 确定性执行层 (Code-as-Action & Sandboxing)          │
            │    • Python / DuckDB 内存数据表执行引擎                │
            │    • 隔离微虚拟机/容器沙箱 (断网/只读挂载/超时强杀)    │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ 4. 认知四层分立记忆系统 (Hierarchical Memory)          │
            │    • Working Memory: 案头活跃草稿纸 (Prompt/REPL)      │
            │    • Episodic Memory: 专案因果侦查日志 (State Log)     │
            │    • Semantic Memory: 会计法规与行业基准 (Vector/Graph)│
            │    • Procedural Memory: 现成审计代码技能库 (.py)       │
            └────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │ 5. 质量保证与评测系统 (Dual-Track Evals & Harness)     │
            │    • 轨 1: 绝对容差代码直判 (|计算值 - 真值| <= 0.1%)   │
            │    • 轨 2: 协议去偏裁判 (双盲位置翻转 Swap Bias Test)  │
            │    • 自动化评测架 (Harness): 断点续跑、考场防污染、超时 │
            └────────────────────────────────────────────────────────┘
```

---

## 二、项目目录结构规范

```text
D:\gemini folder\finreason-agent\
├── MASTER_SPEC.md              # 本项目主宪章与架构蓝图
├── pyproject.toml              # 现代依赖管理配置
├── README.md                   # 项目说明与工业快速起手指南
├── configs/                    # 配置文件（模型接入、沙箱配额、评测参数）
│   └── default.yaml
├── finreason/                  # 核心源代码包
│   ├── core/                   # 大脑与路由规划
│   │   ├── router.py           # 门房认知路由器 (Tier 1/2/3)
│   │   ├── tot_engine.py       # ToT 树状探索与剪枝引擎
│   │   └── mcts_engine.py      # MCTS 终局模拟与回传
│   ├── sandbox/                # 确定性执行与沙箱
│   │   ├── code_runner.py      # Python / DuckDB 解释器
│   │   └── container_box.py    # 隔离容器/环境运行守卫
│   ├── memory/                 # 四层记忆体系
│   │   ├── working.py          # 工作记忆
│   │   ├── episodic.py         # 专案时序日志与 Reflexion Buffer
│   │   ├── semantic.py         # 语义记忆检索器
│   │   └── procedural.py       # 程序记忆代码注册表
│   ├── parser/                 # 财报与文档对齐
│   │   ├── pdf_parser.py       # 视觉版面表格提取
│   │   ├── chunker.py          # 父子层级切块
│   │   └── footnote_linker.py  # 主表与附注双向锚定器
│   └── evals/                  # 第四代双轨评测体系
│       ├── harness.py          # 自动化考场框架 (超时/隔离/断点)
│       ├── deterministic.py    # 轨 1 代码绝对容差判分
│       └── debiased_judge.py   # 轨 2 双盲去偏裁判
├── data/
│   ├── raw/                    # 原始待审年报 PDF (康美药业、茅台等)
│   └── processed/              # 解析后的结构化 DuckDB 数据库与 Markdown
└── tests/                      # 单元测试与集成冒烟测试
```

---

## 三、三阶段实战交付路线图

### 第一阶段（第 1~4 周）：确定性骨架打通
- [x] 搭建环境与依赖 (`pyproject.toml`)
- [x] 实现 `sandbox/code_runner.py`（零算术幻觉 DuckDB 计算核心）
- [x] 实现 `core/router.py`（基于受限解码的 Tier 1/2/3 认知分流）
- [x] 验证：在本地沙箱里，零错误核算出三表基础勾稽平衡

### 第二阶段（第 5~8 周）：文档穿透与刑侦推理
- [x] 攻克 `parser/`：提取真实 PDF 财报跨页大表，完成附注外键绑定
- [x] 攻克 `memory/`：四层记忆分工协作，支持 Reflexion 回溯
- [x] 攻克 `core/tot_engine.py`：构建刑侦假设树，自动顺藤摸瓜找证据
- [x] 搭建 `evals/harness.py`：录入首批 20 个真实暴雷/健康案例跑初测

### 第三阶段（第 9~12 周）：基准压测、微调提效与影响力收官
- [x] 实现 `core/mcts_engine.py`：蒙特卡洛刑侦终局模拟与价值回传（Selection, Expansion, Rollout, Backpropagation）
- [x] 扩充黄金评测集至 100 案（50 真实暴雷 + 50 真实健康），完成端到端第四代双轨测评大屏（100% 满分通关）
- [x] 实现 `evals/trajectory_exporter.py`：导出 100 案 Alpaca SFT、ShareGPT 多轮对话与 DPO 偏好对数据集，赋能开源 7B/14B 小模型微调
- [x] 撰写英文技术报告 `TECHNICAL_REPORT.md` 与交互式实战演示 `run_demo.py`，完成全球前 1% 金融刑侦 Agent 架构收官！

