FROM tensorflow/serving:latest

COPY model_trained/export/exporter/ /model

EXPOSE 9000

ENTRYPOINT tensorflow_model_server --rest_api_port=9000 --model_name=default --model_base_path=/model/