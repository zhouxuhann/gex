FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# 复制代码
COPY src/ src/
COPY config/ config/

# 创建数据目录
RUN mkdir -p /data

# 暴露端口
EXPOSE 8050

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# 启动命令
CMD ["python", "-m", "gex_monitor.main", "--config", "/app/config/config.yaml"]
