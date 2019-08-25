#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, url_for, send_from_directory, request, jsonify
import logging
import os
import shutil
import requests
import json
import uuid
from werkzeug import secure_filename
from scripts_keras import process_data

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/Diseases/Testing/1'.format(PROJECT_HOME)
TRAIN_UPLOAD_FOLDER = '{}/Diseases/Training'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRAIN_UPLOAD_FOLDER'] = TRAIN_UPLOAD_FOLDER


def create_new_folder(local_dir):
    shutil.rmtree(local_dir)
    os.makedirs(local_dir)
    return local_dir


@app.route('/', methods=['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)
        app.logger.info(img_name)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info('saving {}'.format(saved_path))
        img.save(saved_path)
        (data, percent) = process_data()
        if percent >= 95:
            path = os.path.join((TRAIN_UPLOAD_FOLDER+'/'+data.get('id')).format(PROJECT_HOME), str(uuid.uuid4())+img_name)
            print(path)
            img.save(path)
        return jsonify(data)
    else:
        return 'Where is the image?'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)

			
