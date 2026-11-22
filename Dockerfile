# Blast Radius selector. Same base image the commits were labelled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# The packages sqlglot's tests were labelled with (PyPI only).
COPY requirements.lock /app/requirements.lock
RUN pip install -r /app/requirements.lock

# sqlglot at eb11aec4be8cb2122afe713a4816662c3f0fb2fa (git archive of that commit).
COPY vendor/sqlglot-eb11aec.tar.xz /tmp/sqlglot.tar.xz
RUN python -c "import tarfile; tarfile.open('/tmp/sqlglot.tar.xz').extractall('/app/src', filter='data')" \
 && rm /tmp/sqlglot.tar.xz

COPY brlib.py brfp.py brpool.py brreplay.py /app/
COPY build/record_replay.py /app/build/
COPY maps/maps.pkl.xz /app/maps/maps.pkl.xz

# Bytecode for the pristine tree (the run's disk is read-only), then the
# pristine fingerprints of every module global and class attribute, computed
# by this very interpreter.
RUN python -c "import lzma, shutil; shutil.copyfileobj(lzma.open('/app/maps/maps.pkl.xz'), open('/app/maps/maps.pkl', 'wb'))" \
 && rm /app/maps/maps.pkl.xz \
 && cd /app/src \
 && python -m compileall -q -j 4 sqlglot tests \
 && BR_MAPS_PKL=/app/maps/maps.pkl PYTHONPATH=/app/src python /app/brfp.py /app/maps/fp.pkl 1 2 3 4 5 \
 && (PYTHONPATH=/app/src python /app/build/record_replay.py /app/src /app/maps/replay.pkl || rm -f /app/maps/replay.pkl)

# Comparison records from the pinned suite. Shipping the completed map
# keeps fresh image builds independent of the recorder's time budget.
COPY maps/values.tar.xz /tmp/values.tar.xz
RUN python -c "import tarfile; tarfile.open('/tmp/values.tar.xz').extractall('/app/maps', filter='data')" \
 && rm /tmp/values.tar.xz

COPY selector.py /app/
RUN python -m compileall -q /app/selector.py /app/brlib.py /app/brfp.py /app/brpool.py /app/brreplay.py \
 && chmod -R a+rX /app

CMD ["python", "/app/selector.py"]
