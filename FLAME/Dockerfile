FROM python:3.8

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/soubhiksanyal/FLAME_PyTorch
WORKDIR /FLAME_PyTorch
RUN python -m pip install --upgrade pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN pip install flask
RUN python setup.py install

COPY model /FLAME_PyTorch/model
COPY Flame_Modules/app.py Flame_Modules/getFlame.py /FLAME_PyTorch/

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["app.py"]