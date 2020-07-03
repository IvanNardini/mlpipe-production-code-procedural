ARG INPUT_IMAGE
ARG TAG
# Container to run the command for image building
FROM alpine:3.10

ENV INPUT_IMAGE $INPUT_IMAGE
ENV TAG $TAG

# Copy file for building
COPY . .

RUN apk add --update --no-cache docker
RUN ["chmod", "+x", "/build.sh"]

# build image when container starts
ENTRYPOINT "/build.sh" ${INPUT_IMAGE} ${TAG}