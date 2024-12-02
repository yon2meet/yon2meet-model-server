import io
import os

from dotenv import load_dotenv
from flask import Flask, send_file, request

from model import model

load_dotenv()

SERVER_PORT = int(os.environ.get('SERVER_PORT'))

application = Flask(__name__)


@application.post("/image")
def image():
    prompt = request.json["prompt"]  # TODO currently file name

    image_bytes = model(prompt)

    mem = io.BytesIO()
    mem.write(image_bytes)
    mem.seek(0)  # seeking was necessary. Python 3.5.2, Flask 0.12.2

    return send_file(
        mem,
        as_attachment=True,
        download_name=prompt,
        mimetype='image/jpeg'
    )


if __name__ == "__main__":
    application.run(port=SERVER_PORT)
