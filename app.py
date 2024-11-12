from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Renders the form page (index.html)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if 'image' is part of the request
    if 'image' not in request.files:
        return "No file uploaded", 400

    image = request.files['image']
    
    # Get the percentage value from the form
    percentage = request.form.get('percentage')

    if not percentage:
        return "Percentage value is missing", 400

    try:
        np_components = float(percentage)
    except ValueError:
        return "Invalid percentage value. Please enter a numeric value.", 400

    if np_components <= 0 or np_components > 1:
        return "Percentage value must be between 0 and 1", 400

    if image.filename == '':
        return "No file selected", 400

    if image and allowed_file(image.filename):
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Process the image
        processed_image = reduce_image(image_path, np_components)

        # Render the success page with the processed image filename and confidence percentage
        return render_template('success.html', confidence=np_components * 100, processed_image=processed_image)

    return "Invalid file type. Only .jpg, .jpeg, .png, .gif are allowed", 400

def reduce_image(file_name, np_components):
    # Step 1: Load the original image
    image = io.imread(file_name)
    gray_image = color.rgb2gray(image)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=np_components)
    transformed_image = pca.fit_transform(gray_image)

    # Step 3: Reconstruct the compressed image
    reconstructed_image = pca.inverse_transform(transformed_image)

    # Normalize and convert to uint8
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)

    # Save processed image
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_image.jpg')
    io.imsave(processed_image_path, compressed_image_uint8)

    # Return the filename of the processed image
    return 'compressed_image.jpg'

@app.route('/uploads/<filename>')
def download(filename):
    # Serve the file from the UPLOAD_FOLDER
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)
