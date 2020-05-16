#lightweight python
from python:3.7-slim

# Copy local code to the container image
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

#Install Dependencies
RUN pip install numpy tensorflow==2.1.0 Flask gunicorn google-cloud-logging

#Run the flask service on container
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 SAGunicorn:app