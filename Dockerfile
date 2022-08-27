FROM python:3.9.7
RUN mkdir /code
COPY requirements.txt /code
RUN pip install -r /code/requirements.txt
COPY . /code
RUN apt update
RUN apt install -y gettext
WORKDIR /code
ENV FOLDER='output_folder'
CMD python baseline.py --folder ${FOLDER}
