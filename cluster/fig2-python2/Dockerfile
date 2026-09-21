# syntax=docker/dockerfile:1

# Miniconda2 is EOL, but this immutable amd64 image provides a reproducible
# Python 2.7 runtime without mixing Ubuntu 24.04 system libraries into the
# historical scientific stack.
FROM continuumio/miniconda2:4.7.12@sha256:c1c84747f3ed7c4eca1f59c2c2ea954565249853b791b0c8fbf9fc866b715f4f

ARG SORN_COMMIT=cdad55d55f39e04f568ca1bc0c6036bec8db08fb

LABEL org.opencontainers.image.source="https://github.com/delpapa/SORN" \
      org.opencontainers.image.revision="${SORN_COMMIT}" \
      org.opencontainers.image.description="Original Python 2 SORN, Fig. 2 N_e=200 simulation"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# These match the historical Python 2 environment used for the local
# differential validation and current original-code Fig. 2 simulations.
RUN conda install --yes \
        numpy=1.16.6 \
        matplotlib=2.2.3 \
        pytables=3.5.2 \
    && pip install --no-cache-dir --only-binary=:all: \
        scipy==1.2.3 \
    && pip install --no-cache-dir \
        ipdb==0.13.9 \
        powerlaw==1.4.6 \
    && conda clean --all --yes

# Fetch a fresh archive of the author's repository at the exact commit used
# in the Python 2 comparison. No files are copied from the local Python 3 port.
RUN mkdir -p /opt/sorn \
    && wget --quiet --output-document=/tmp/sorn.tar.gz \
       "https://github.com/delpapa/SORN/archive/${SORN_COMMIT}.tar.gz" \
    && tar --extract --gzip --file=/tmp/sorn.tar.gz \
       --directory=/opt/sorn --strip-components=1 \
    && rm /tmp/sorn.tar.gz

# Two cluster-safety patches only:
# 1. SORN_SEED prevents simultaneously launched jobs from receiving the same
#    wall-clock-second seed. With no SORN_SEED, original behavior is preserved.
# 2. SORN_RUN_ID gives every job a separate directory on a shared volume.
RUN sed -i \
      "s/start_seed = current_time_int\*(rank + 1)/start_seed = int(os.environ.get('SORN_SEED', current_time_int*(rank + 1)))/" \
      /opt/sorn/utils/backup.py \
    && sed -i \
      's#utils.initialise_backup(mount="../", dest="../backup")#utils.initialise_backup(mount="../", dest=os.path.join("../backup", os.environ.get("SORN_RUN_ID", "manual")))#' \
      /opt/sorn/common/test_single.py \
    && grep -Fq "start_seed = int(os.environ.get('SORN_SEED'" /opt/sorn/utils/backup.py \
    && grep -Fq 'c.N_e = 200' /opt/sorn/delpapa/param_Zheng2013.py \
    && grep -Fq 'c.steps_plastic = 5000000' /opt/sorn/delpapa/param_Zheng2013.py

COPY entrypoint.sh /usr/local/bin/sorn-fig2
RUN chmod 0755 /usr/local/bin/sorn-fig2 \
    && python -c "import numpy, scipy, tables, matplotlib, powerlaw; print('Python 2 scientific stack validated')"

WORKDIR /opt/sorn/common
VOLUME ["/opt/sorn/backup"]

ENTRYPOINT ["/usr/local/bin/sorn-fig2"]
