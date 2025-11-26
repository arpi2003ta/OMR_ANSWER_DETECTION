from flask import Flask, request, jsonify, render_template,send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
from omr_processor import process_omr

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
SUBJECT_FOLDER = r'img\subjects'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUBJECT_FOLDER, exist_ok=True)

app.config['SUBJECT_FOLDER'] = SUBJECT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_uploads_folder():
    """Remove all previous uploaded files before saving a new one"""
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

@app.errorhandler(RequestEntityTooLarge)
def handle_big_file(e):
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 16MB."
    }), 413

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_omr', methods=['POST'])
def handle_omr():
    try:
        if 'omr_image' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['omr_image']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

        # Remove old uploads before saving new one
        clear_uploads_folder()

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f'Saving file to {image_path}')
        file.save(image_path)

        # Process OMR
        results = process_omr(image_path)
        print(f'results', results)
  
        if results['success']:
            return jsonify({
                'success': True,
                'answers': results['answers'],
                'subject_scores': results['subject_scores'],
                'total_answers': results['total_answers'],
                'filename': filename,
                'subject_images': results['subject_images']
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': results.get('error', 'Processing failed')
            }), 500

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/subjects/<path:filename>')
def get_subject_image(filename):
    return send_from_directory(app.config['SUBJECT_FOLDER'], filename)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
