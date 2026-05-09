# ---------- Stage 1: Builder ----------
FROM python:3.12-slim as builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy Project 
COPY pyproject.toml .

#Install Dependencies
RUN uv pip install --system --prefix=/install .

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim

WORKDIR /app

# Copy only installed dependencies from builder
COPY --from=builder /install /usr/local

COPY RAG/ .

# Expose LangGraph Dev port
EXPOSE 8123

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8123"]