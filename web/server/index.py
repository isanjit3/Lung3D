import os
import uuid

from flask import Flask, jsonify,request,url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('You are connected to Flask API!!!')

#Upload
@app.route('/upload',methods=['POST'])
def uploadFile():
    if request.method == 'POST':

        file = request.files['file']
        filename = secure_filename(file.filename)

        # Gen GUUID File Name
        fileExt = filename.split('.')[1]
        autoGenFileName = uuid.uuid4()

        newFileName = str(autoGenFileName) + '.' + fileExt
        saveFile = os.path.join(app.config['UPLOAD_FOLDER'], newFileName)
        if not os.path.exists(os.path.dirname(saveFile)):
            os.makedirs(os.path.dirname(saveFile))
            
        file.save(saveFile)
      
        return jsonify("File successfully uploaded.")


if __name__ == '__main__':
    app.run(host='192.168.86.43', port = 5000)