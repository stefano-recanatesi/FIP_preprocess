# hash:sha256:c0513ecef6b5c48988e13905c6ece91b2f8ceca19ba22b87aa4b7c3f131e7eb2
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    aind-data-schema \
    jupyterlab \
    matplotlib \
    notebook \
    numpy \
    pandas \
    pynwb \
    scikit-learn \
    scipy \
    seaborn

COPY postInstall /
RUN /postInstall
