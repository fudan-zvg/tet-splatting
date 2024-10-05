##!/usr/bin/env bash

# wget https://drive.switch.ch/index.php/s/RFfTZwyKROKKx0l/download
# unzip -j download -d pretrained_models
# rm download

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" |  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update -y && apt-get install -y google-cloud-sdk

apt install -y imagemagick

pip install gdown
mkdir -p pretrained_models

# https://drive.google.com/uc?id=1UxUDbEygQ-CMBjRKACw_Xdj4RkDjirB5&confirm=t
gdown '1UxUDbEygQ-CMBjRKACw_Xdj4RkDjirB5&confirm=t' -O ./pretrained_models/ # omnidata depth (v1)
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' -O ./pretrained_models/ # omnidata depth (v2)
