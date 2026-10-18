FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
WORKDIR /app
RUN useradd --create-home --uid 10001 app
COPY pyproject.toml ./
COPY backend ./backend
RUN pip install --no-cache-dir .
COPY public ./public
USER app
EXPOSE 8000
CMD ["python", "-m", "backend.main"]
