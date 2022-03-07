FROM tensorflow/tensorflow:2.1.0-gpu-py3 as builder

ENV LC_ALL C.UTF-8
ENV TZ=Europe/Berlin

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  NUMBA_CACHE_DIR=/tmp \
  POETRY_VIRTUALENVS_CREATE=false \
  OMP_NUM_THREADS=4 \
  NUMBA_NUM_THREADS=4 \
  PATH="$PATH:$HOME/.poetry/bin"

RUN apt update && apt-get install -y git && apt-get clean

RUN mkdir /data
WORKDIR /data
RUN ulimit -c 0

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

COPY ./pyproject.toml ./poetry.lock /data/
RUN $HOME/.poetry/bin/poetry install --no-dev --no-interaction --no-root

FROM builder
COPY ./ /data/
RUN $HOME/.poetry/bin/poetry install --no-dev --no-interaction
