#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "          FinReason-Agent 容器化一键部署与启动脚本 (Linux/macOS)          "
echo "======================================================================"

if ! command -v docker &> /dev/null; then
    echo "[ERROR] 未检测到 Docker 环境，请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo ""
echo "[1/3] 正在构建 FinReason-Agent 镜像并启动容器集群..."
docker compose up -d --build

echo ""
echo "[2/3] 等待服务初始化与健康检查..."
sleep 5
docker compose ps

echo ""
echo "[3/3] 部署成功！访问入口："
echo "----------------------------------------------------------------------"
echo " - 🌐 可视化审计大屏 (Streamlit): http://localhost:8501"
echo " - ⚡ 后端 RESTful API 根路径:   http://localhost:8000"
echo " - 📚 OpenAPI / Swagger 在线文档: http://localhost:8000/docs"
echo " - 📖 ReDoc 交互式技术文档:       http://localhost:8000/redoc"
echo "----------------------------------------------------------------------"
echo ""
