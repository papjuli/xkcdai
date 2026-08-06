# Public HTTP build of the xkcdai MCP server, for hosting as a Claude custom connector.
# The prebuilt index (data/) and the embedding model are baked in, so the container
# starts fast and never re-scrapes xkcd/explainxkcd.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    XKCDAI_TRANSPORT=streamable-http \
    XKCDAI_DATA_DIR=/app/data \
    FASTEMBED_CACHE_DIR=/app/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PORT=8000

# A container sees the *host's* core count, so these pools would otherwise be
# sized for a machine we don't have. 
# Serving is one 3269x384 matrix-vector product and one embedding per query, 
# so a single thread is plenty. XKCDAI_ORT_THREADS covers ONNX Runtime, 
# which uses its own pool rather than OpenMP; the rest cover numpy/BLAS. 
# Set here and not in the code so that `xkcdai build` can use all the cores 
# it can get.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    XKCDAI_ORT_THREADS=1

WORKDIR /app

# Apply available OS security patches on top of the base image, then drop apt's
# lists to keep the layer small. (Rebuild with `docker build --pull` periodically
# so this runs against a freshly-patched base.)
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Install the package (deps cached as their own layer).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Ship the prebuilt index (run `xkcdai build && xkcdai enrich && xkcdai build`
# locally first so data/ exists). data/ is gitignored but lives in the build context.
COPY data ./data

# Bake the embedding model into the image so cold starts don't download it.
RUN python -c "from xkcdai.embed import _get_model; _get_model()"

EXPOSE 8000
CMD ["xkcdai-server"]
