FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

LABEL maintainer="ruihirano"

COPY . /workspace

# for opencv error
RUN apt-get update && apt-get install -y libgtk2.0-dev

RUN pip install --upgrade pip 
RUN pip install -r requirements.txt 

