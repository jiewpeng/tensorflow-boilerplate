FROM tensorflow/serving:latest-devel as build_image

FROM ubuntu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server
COPY model_trained/export/exporter/ /model

EXPOSE 9000

ENTRYPOINT tensorflow_model_server --rest_api_port=9000 --model_name=default --model_base_path=/model/