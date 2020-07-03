# Container to run the command for image building
FROM alpine:latest

# Copy file for building
COPY . .

RUN apk add --update --no-cache docker
RUN ["chmod", "+x", "/build.sh"]

# build image when container starts
ENTRYPOINT "/build.sh"