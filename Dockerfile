FROM python:3.12-slim-bookworm AS build

COPY . /src

RUN python3 -m venv /app/venv && PATH=/app/venv/bin:$PATH pip install /src

FROM python:3.12-slim-bookworm AS final

COPY --from=build /app/venv /app/venv/
COPY pv2mqtt.py /app/pv2mqtt.py

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:${PATH}"

ENTRYPOINT [ "python3", "/app/pv2mqtt.py" ]
CMD [ "/pv2mqtt.yml" ]
