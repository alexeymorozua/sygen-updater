# Sygen updater sidecar — checks GHCR for new image digests and exposes
# an authenticated /apply endpoint that runs `docker compose pull && up -d`
# on the host's docker socket.
FROM python:3.12-alpine

# docker-compose v2 plugin is not bundled in this image; the updater
# shells out to the host's docker binary via the mounted socket, so we
# install the minimal docker CLI + compose plugin inside the container.
RUN apk add --no-cache docker-cli docker-cli-compose tzdata

WORKDIR /app

RUN pip install --no-cache-dir \
        "fastapi>=0.110,<1.0" \
        "uvicorn[standard]>=0.27,<1.0" \
        "httpx>=0.27,<1.0"

COPY updater.py /app/updater.py

EXPOSE 8082

CMD ["uvicorn", "updater:app", "--host", "0.0.0.0", "--port", "8082", "--log-level", "info"]
