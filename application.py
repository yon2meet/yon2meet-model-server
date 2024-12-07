import os

from dotenv import load_dotenv
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS

from timetable_analyzer import analyze_timetable

load_dotenv()

SERVER_PORT = int(os.environ.get('SERVER_PORT'))

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = './uploads'

CORS(application)


@application.route("/image", methods=['POST'])
def image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    schedule = analyze_timetable(path)

    result = sorted(schedule, key=lambda x: (x['day'], x['time']))

    return {"schedule": result}


if __name__ == "__main__":
    application.run(port=SERVER_PORT)
