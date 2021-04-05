FROM python:3.8

#RUN apt-get update && apt-get install -y python-rdkit librdkit1 rdkit-data vim cmake
RUN apt-get update && apt-get install -y vim cmake

WORKDIR /app

COPY . . 

#RUN tar -xvf rdkit/rdkit.tar.gz -C /tmp

RUN pip install -r requirements.txt

