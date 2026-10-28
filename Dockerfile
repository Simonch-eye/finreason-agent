# ==============================================================================
# FinReason-Agent 生产级容器化构建镜像 (Production Dockerfile)
# 支持 FastAPI RESTful 微服务与 Streamlit 可视化审计工作台双模运行
# ==============================================================================

FROM python:3.11-slim AS base

# 环境变量设置：禁用 Python 缓存写入与缓冲输出，确保日志实时打印
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 安装必要的系统运行时工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖配置并预安装 Python 依赖库
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e .

# 复制整个项目源代码
COPY . .

# 预创建持久化数据与日志目录
RUN mkdir -p data/raw data/processed data/checkpoints logs

# 暴露端口：8000 (FastAPI RESTful API), 8501 (Streamlit 可视化大屏)
EXPOSE 8000 8501

# 健康检查探针
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认启动命令：启动生产级 FastAPI RESTful 服务
CMD ["finreason", "serve", "--host", "0.0.0.0", "--port", "8000"]
