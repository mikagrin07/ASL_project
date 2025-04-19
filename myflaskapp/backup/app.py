from turtle import width
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import os
from Project_YA_mika import *  # Import your model and layer classes



app = Flask(__name__, static_url_path='', static_folder='.')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])


def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    print (f"file: {file}")
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
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
        print (f"width: {width}")
        print (f"height: {height}")
        image64 = np.array(image64)
        my_image = image64.reshape(g_num_px*g_num_px,1)
        my_image = my_image/255. - 0.5
        p = curr_model.predict(my_image) 
        prediction_index = np.argmax(p, axis=0)
        result = chr(ord('a') + int(prediction_index))


        return jsonify({'predict' : result, 'width': width, 'height': height, 'url': f'/uploads/{file.filename}'})
    return jsonify({'error': 'File upload failed'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
