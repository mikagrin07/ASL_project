from flask import Flask, request, jsonify, send_from_directory, send_file, render_template
import os
import cProfile
import pstats
import io
import sys
from generate_joints_and_classification import generate_joints

app = Flask(__name__, static_url_path='', static_folder='.')

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_videos'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def root():
    return app.send_static_file('index_mp4.html')


def profile_function(func):
    """Decorator to profile a function using cProfile."""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()  # Start profiling
        result = func(*args, **kwargs)
        profiler.disable()  # Stop profiling

        # Capture profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(50)  # Show top 10 slowest functions

        print("\n--- Profiling Results ---")
        print(s.getvalue())
        print("--- End of Profiling ---\n")

        return result

    return wrapper

@app.route('/upload', methods=['POST'])
#@profile_function  # Applying the profiler
def upload():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        output_filename = f"processed_{file.filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

        # Save uploaded file
        try:
            file.save(input_path)

            # Process video and classify
            predicted_sign = generate_joints(input_path, output_path, PROCESSED_FOLDER)

            return jsonify({
                'url': f'/processed_videos/{output_filename}',
                'predicted_sign': predicted_sign
            })

        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file format. Only MP4 allowed!'}), 400

@app.route('/processed_videos/<filename>')
def processed_video(filename):
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({'error': 'Video file not found'}), 404

if __name__ == '__main__':
    print(sys.executable)
    app.run(debug=True)
