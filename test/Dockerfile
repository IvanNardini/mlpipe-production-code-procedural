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

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

WORKDIR /usr/src

COPY . . 

RUN chmod +x ./run.sh

# "/bin/bash", "./run.sh", "train", "score"

CMD ["/bin/bash", "./run.sh", "train", "score"]