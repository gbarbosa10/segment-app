FROM python:3.11-slim

WORKDIR /segment_app

COPY ./requirements.txt /segment_app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /segment_app/requirements.txt

COPY ./app/ /segment_app/app/

COPY ./static/ /segment_app/static/

EXPOSE 8000

CMD ["uvicorn", "app.main:application", "--host", "127.0.0.1", "--port", "8000"]
