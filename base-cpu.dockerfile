FROM python:3

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/grip-unina/TruFor.git

# Model

# CPU-only installation to save space
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install -q timm==0.6.13 numpy pillow scikit-learn matplotlib tqdm yacs>=0.1.8

WORKDIR /usr/src/app/TruFor



RUN wget -q -c https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
RUN unzip -q -n TruFor_weights.zip && rm TruFor_weights.zip

WORKDIR /usr/src/app/TruFor/TruFor_train_test
