# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ────────────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL description="Email Triage Reinforcement Learning Simulation"

# ── Environment variables ───────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_BASE_URL="https://api.openai.com/v1" \
    MODEL_NAME="gpt-4o-mini" \
    HF_TOKEN="dummy"

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source files ─────────────────────────────────────────────────────────
COPY env.py .
COPY inference.py .

# ── Default command ───────────────────────────────────────────────────────────
# Runs all three difficulty levels with 1 episode each.
# Override via: docker run --rm email-triage-rl --task easy --episodes 5
ENTRYPOINT ["python", "inference.py"]
