# Dockerfile of Example
# Version 1.0
# Base Images
FROM registry.cn-shanghai.aliyuncs.com/aliseccompetition/tensorflow:1.1.0-devel-gpu
#MAINTAINER
MAINTAINER AlibabaSec

ADD . /competition

WORKDIR /competition
RUN pip --no-cache-dir install  -r requirements.txt

RUN mkdir ./mymodels
RUN curl -O 'http://ijcai-yq-hangzhou.oss-cn-hangzhou-internal.aliyuncs.com/defensemodels.tar.gz' && tar -xvf defensemodels.tar.gz -C ./mymodels/ && rm defensemodels.tar.gz
