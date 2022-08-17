UID := $(shell id -u)
GID := $(shell id -g)

datagen: .PHONY
	mkdir data
	UID=$(UID) GID=$(GID) docker compose run datagen

build_notebook:
	docker build -t seal-notebook -f Dockerfile.notebook .

run_notebook:
	mkdir -p notebooks data
	docker run \
        -e JUPYTER_RUNTIME_DIR=/runtime \
        -e JUPYTER_DATA_DIR=/runtime \
        -e JUPYTER_CONFIG_DIR=/runtime \
        -e MPLCONFIGDIR=/runtime \
        --user $(UID):$(GID) \
        -v $(PWD)/notebooks:/notebooks \
        -v $(PWD)/data:/data:rw \
        -p 8888:8888 \
        -it seal-notebook

.PHONY:
