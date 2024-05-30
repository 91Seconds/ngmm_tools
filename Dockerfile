FROM python:3.12-bookworm AS base

ENV VENV_PATH /poetry
ENV POETRY ${VENV_PATH}/bin/poetry

RUN python3 -m venv $VENV_PATH && \
    $VENV_PATH/bin/pip install -U pip setuptools && \
    $VENV_PATH/bin/pip install poetry

COPY pyproject.toml .
COPY poetry.lock .
COPY README.md .

RUN .${POETRY} install --no-root

COPY Analyses Analyses

RUN .${POETRY} install
