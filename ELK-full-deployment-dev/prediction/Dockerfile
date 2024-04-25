FROM bitnami/pytorch:2

ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt 
WORKDIR /home/

USER root

RUN apt-get update \
&& apt-get -y install nginx \
&& apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/list*

RUN python3 -c "from transformers import DistilBertTokenizerFast; DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')"

COPY predict.py /home/predict.py
COPY CustomModel.py /home/CustomModel.py
COPY entrypoint.sh /home/entrypoint.sh

ENTRYPOINT ["/bin/sh", "/home/entrypoint.sh"]