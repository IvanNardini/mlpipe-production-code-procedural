FROM ubuntu:18.04

MAINTAINER  Ivan Nardini "ivan.nardini@sas.com"

RUN apt-get update && \
	apt-get install -y sudo \
	build-essential curl \
	libcurl4-openssl-dev \
	libssl-dev wget \
	python3-pip \
	git \
	ruamel.yaml \
	graphviz && \
	pip3 install --upgrade pip

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY insurance_claims.csv insurance_claims.csv
COPY preprocess.py preprocess.py

CMD [ "python", "./preprocess.py" ]