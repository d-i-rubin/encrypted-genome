UID := $(shell id -u)
GID := $(shell id -g)

datagen: .PHONY
	mkdir data
	UID=$(UID) GID=$(GID) docker compose run datagen

notebook: .PHONY
	UID=$(UID) GID=$(GID) docker compose up notebook

stop: .PHONY
	UID=$(UID) GID=$(GID) docker compose down

.PHONY:
