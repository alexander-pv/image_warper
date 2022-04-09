
FROM python:3.9-slim

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    rm -rf /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoremove

WORKDIR /app
COPY src /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501

ENTRYPOINT ["/bin/bash"]
