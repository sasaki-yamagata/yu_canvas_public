FROM continuumio/miniconda3:latest

RUN conda update -y conda \
    && conda install -c conda-forge flask \
    rdkit \
    scikit-learn \
    openbabel \
    networkx && \
    conda clean -i -t -y

ENV PORT 5000

ENV PYTHONPATH /app/module

WORKDIR /app

COPY ./app/ /app

CMD ["python", "main.py"]
