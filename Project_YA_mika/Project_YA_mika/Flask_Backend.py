
# A very simple Flask Hello World app for you to get started with...

#from flask import Flask

#app = Flask(__name__)

#@app.route('/')
#def hello_world():
#    return 'Hello from Flask!'

###########

from turtle import width
from flask import Flask, request, jsonify, send_from_directory, send_file
from PIL import Image
import numpy as np
import os
from Project_YA_mika import *  # Import your model and layer classes
from pathlib import Path


app = Flask(__name__, static_url_path='', static_folder='.')

UPLOAD_FOLDER = 'uploads'
#UPLOAD_FOLDER2 = 'myflaskapp//uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#ppp = app.config['UPLOAD_FOLDER']
print (f"upload: { app.config['UPLOAD_FOLDER']}")


@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])


def upload():
    print ("upload start --------------------------------")

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    print (f"file: {file}")

    print ("upload start -------------------------------- #########################################")
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join("myflaskapp",UPLOAD_FOLDER, file.filename)

        print (f"filepath: {filepath}")
        file.save(filepath)

        # Process the image
        img = Image.open(filepath)
        width, height = img.size
        print (f"width: {width}")
        print (f"height: {height}")


        ########### Load the model and weights here ###########
        curr_model = create_signs_model()
        curr_model.forced_num_classess = g_subset_num_classes


        #img = img.convert('RGB')
        img = img.convert('L')
        image64 = img.resize((g_num_px, g_num_px), )
        width, height = image64.size
        print (f"width: {width}, height: {height}")
       # print (f"height: {height}")
        image64 = np.array(image64)
        my_image = image64.reshape(g_num_px*g_num_px,1)
        my_image = my_image/255. - 0.5
        p = curr_model.predict(my_image)
        print (f"Prediction =  {p}  ------------------------------")
        prediction_index = np.argmax(p, axis=0)
        result = chr(ord('a') + int(prediction_index))
        print (f"upload end result {result}--------------------------------")

        return jsonify({'predict' : result, 'width': width, 'height': height, 'url': f'/uploads/{file.filename}'})
    return jsonify({'error': 'File upload failed'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print (f"download start +++++++++++++++++++++++++++++ filename={filename}")
    #return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    fileToDownload = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.isfile(fileToDownload):
        print(f"File {fileToDownload} exists.")
    else:
        print(f"File {fileToDownload} doesn't exist.")
    root = Path('.')
   # folder_path = root + UPLOAD_FOLDER;
   # print(f"folder_path: {folder_path}")
   #fullFilePath = os.path.abspath(fileToDownload)
   #return send_file(fullFilePath, as_attachment=True)
    return send_from_directory(UPLOAD_FOLDER, filename)
    #return send_file(fileToDownload)
    #return send_from_directory('/myflaskapp/uploads', filename)




if __name__ == '__main__':
    app.run(debug=True)

