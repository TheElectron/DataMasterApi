FROM python:3.8
RUN apt-get update && apt-get upgrade -y && apt-get install -y
RUN apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
WORKDIR /flask
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY /src .
COPY /model ./model
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]