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

WORKDIR /usr/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . . 

RUN chmod +x ./run.sh

# COPY . .

# CMD [ "python3", "./train.py" ]

CMD ["/bin/bash", "./run.sh", "train", "score"]