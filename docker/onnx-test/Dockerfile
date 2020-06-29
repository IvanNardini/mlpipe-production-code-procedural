FROM python:3

MAINTAINER  Ivan Nardini "ivan.nardini@sas.com"

RUN apt-get update && \
	apt-get install -y sudo \
	ruamel.yaml \
	graphviz && \
	pip3 install --upgrade pip

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ../../data/raw/*.csv insurance_claims.csv
COPY ../../src/procedural_ml_pipe/preprocess.py preprocess.py

CMD [ "python", "./preprocess.py" ]