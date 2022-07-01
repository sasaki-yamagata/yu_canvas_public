FROM continuumio/miniconda3:latest

RUN conda update -y conda \
    && conda install -c conda-forge flask \
    rdkit \
    scikit-learn=0.24.2

RUN conda install -c conda-forge networkx \
    openbabel

ENV PORT 5000

ENV PYTHONPATH /app/module

WORKDIR /app

COPY ./app/ /app

CMD ["python", "main.py"]
