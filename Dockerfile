FROM pytorchlightning/pytorch_lightning:latest

WORKDIR /ck3-image-portrait-modeling-container
COPY ./requirements.txt .

RUN apt install -y git

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs

RUN pip install -r requirements.txt



