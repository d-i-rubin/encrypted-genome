FROM debian:bullseye

# prevents update and install asking for tz
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt update && \
    apt install -y git build-essential cmake python3 python3-dev python3-pip

ARG SRC=/python-seal
RUN mkdir -p ${SRC}

# update submodules
RUN git clone --recurse https://github.com/ApproximateIdentity/SEAL-Python.git ${SRC}

# install python requirements
RUN pip3 install -r ${SRC}/requirements.txt

# build seal
RUN cd ${SRC}/SEAL && \
    cmake -S . -B build -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF && \
    cmake --build build

# build seal python bindings
RUN cd ${SRC} && \
    python3 setup.py build_ext -i && \
    python3 setup.py install
