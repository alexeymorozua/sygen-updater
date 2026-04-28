# Sygen updater sidecar — checks GHCR for new image digests and exposes
# an authenticated /apply endpoint that runs `docker compose pull && up -d`
# on the host's docker socket.
FROM python:3.12-alpine

# Pull docker-cli + the compose plugin straight from the official
# `docker:27-cli` image instead of `apk add docker-cli` from Alpine
# repos. Alpine's package lags behind upstream CLI by a release or
# two, and at 24.x its API protocol was 1.43 — Colima 0.10+ runs
# Docker Engine 27 which requires API ≥1.44, so the bundled CLI was
# 500'ing on every `docker compose pull` with
#   "client version 1.40 is too old. Minimum supported API
#   version is 1.44, please upgrade your client to a newer version"
# and breaking the in-app Apply-update flow on every fresh macOS
# Colima install. docker:27-cli is itself Alpine-based so the binary
# is musl-compatible with python:3.12-alpine.
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker:27-cli /usr/local/libexec/docker/cli-plugins/docker-compose \
                          /usr/local/libexec/docker/cli-plugins/docker-compose
RUN apk add --no-cache tzdata

WORKDIR /app

RUN pip install --no-cache-dir \
        "fastapi>=0.110,<1.0" \
        "uvicorn[standard]>=0.27,<1.0" \
        "httpx>=0.27,<1.0"

COPY updater.py /app/updater.py

EXPOSE 8082

CMD ["uvicorn", "updater:app", "--host", "0.0.0.0", "--port", "8082", "--log-level", "info"]
