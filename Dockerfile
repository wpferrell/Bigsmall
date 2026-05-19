FROM python:3.11-slim

LABEL org.opencontainers.image.title="BigSmall"
LABEL org.opencontainers.image.description="Lossless neural network weight compression"
LABEL org.opencontainers.image.url="https://github.com/wpferrell/Bigsmall"
LABEL org.opencontainers.image.source="https://github.com/wpferrell/Bigsmall"
LABEL org.opencontainers.image.licenses="Elastic-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir bigsmall

WORKDIR /data

ENTRYPOINT ["bigsmall"]
CMD ["--help"]
