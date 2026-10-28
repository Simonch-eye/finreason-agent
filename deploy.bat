@echo off
chcp 65001 >nul
echo ======================================================================
echo           FinReason-Agent 容器化一键部署与启动脚本 (Windows)           
echo ======================================================================

where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 未检测到 Docker 环境，请先安装 Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo.
echo [1/3] 正在构建 FinReason-Agent 镜像并启动容器群...
docker compose up -d --build

echo.
echo [2/3] 检查服务健康状态...
timeout /t 5 >nul
docker compose ps

echo.
echo [3/3] 部署成功！访问入口：
echo ----------------------------------------------------------------------
echo  - 🌐 可视化审计大屏 (Streamlit): http://localhost:8501
echo  - ⚡ 后端 RESTful API 根路径:   http://localhost:8000
echo  - 📚 OpenAPI / Swagger 在线文档: http://localhost:8000/docs
echo  - 📖 ReDoc 交互式技术文档:       http://localhost:8000/redoc
echo ----------------------------------------------------------------------
echo.
pause
