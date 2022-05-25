FROM continuumio/miniconda3:latest

RUN conda update -y conda \
    && conda install -c conda-forge flask \
    rdkit \
    scikit-learn

ENV PORT 5000

WORKDIR /app

COPY ./app/ /app

CMD ["python", "main.py"]