UID := $(shell id -u)
GID := $(shell id -g)

build_datagen:
	docker build -t seal-datagen -f Dockerfile.datagen .

run_datagen:
	mkdir -p data
	docker run \
        --env-file env \
        --user $(UID):$(GID) \
        -v $(PWD)/data:/data:rw \
        -v $(PWD)/Challenge:/Challenge:ro \
        -it \
         seal-datagen \
        /datagen/run_generate_model_data.sh

build_notebook:
	docker build -t seal-notebook -f Dockerfile.notebook .

run_notebook:
	mkdir -p notebooks data
	docker run \
        --env-file env \
        -e JUPYTER_RUNTIME_DIR=/runtime \
        -e JUPYTER_DATA_DIR=/runtime \
        -e JUPYTER_CONFIG_DIR=/runtime \
        -e MPLCONFIGDIR=/runtime \
        --user $(UID):$(GID) \
        -v $(PWD)/notebooks:/notebooks \
        -v $(PWD)/data:/data:rw \
        -p 8888:8888 \
        -it seal-notebook
