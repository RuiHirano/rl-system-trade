FROM pytorch/pytorch
LABEL maintainer="ruihirano"

COPY . /workspace

RUN pip install --upgrade pip 
RUN pip install -r requirements.txt 

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

