CONTAINER_NAME ?= rag
SERVE_PORT ?= 7777

sh: build
	docker run -it --rm --ulimit core=0 --gpus=all --ipc=host \
	-v $(shell pwd):/app \
	-p $(SERVE_PORT):8080 \
	--name $(CONTAINER_NAME) \
	$(CONTAINER_NAME) bash


build: Dockerfile
	docker build -t $(CONTAINER_NAME) -f $< .