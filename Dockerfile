FROM pytorchlightning/pytorch_lightning:latest

WORKDIR /ck3-image-portrait-modeling-container

COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN apt install -y git
RUN git lfs install

RUN git clone https://github.com/Hakim1625/ck3-image-portrait-modeling.git