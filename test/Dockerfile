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

RUN pip3 install -r requirements.txt

WORKDIR /usr/src/app

COPY . .

CMD [ "python3", "./preprocess.py" ]