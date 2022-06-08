FROM pytorchlightning/pytorch_lightning:latest

WORKDIR /gridai/project
COPY . .

RUN pip install -r requirements.txt