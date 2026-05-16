FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY notebooks ./notebooks

EXPOSE 8000

CMD ["uv", "run", "marimo", "run", "notebooks", "--host", "0.0.0.0", "--port", "8000", "--no-token"]