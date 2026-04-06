FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install openenv-core fastapi uvicorn gradio requests

EXPOSE 7860
EXPOSE 8000

CMD ["bash", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 8000 & python app.py"]
