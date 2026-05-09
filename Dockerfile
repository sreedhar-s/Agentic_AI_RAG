# ---------- Stage 1: Builder ----------
FROM python:3.12-slim as builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy Project 
COPY pyproject.toml .

RUN uv pip install --system --prefix=/install .

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install/. /usr/local/

COPY RAG/ .

EXPOSE 8000

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8000"]