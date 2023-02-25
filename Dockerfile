FROM python:3.11-alpine

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

COPY pv2mqtt.py /app/pv2mqtt.py

ENV PYTHONUNBUFFERED=1
ENTRYPOINT [ "python3", "/app/pv2mqtt.py" ]
CMD [ "/pv2mqtt.yml" ]
