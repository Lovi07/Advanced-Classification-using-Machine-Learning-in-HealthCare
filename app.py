from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import mahotas as mh
from matplotlib import pyplot as plt
from tensorflow.keras.models import model_from_json



app = Flask(__name__)

# Load model and class names
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("model.weights.h5")


with open('lab.pickle', 'rb') as f:
    lab = pickle.load(f)

def diagnosis(image):
    # Prepare image for classification
    IMM_SIZE= 224
    
    # Prepare image to classification
    ##YOUR CODE GOES HERE##
    if len(image.shape) > 2:
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]]) #resize of images RGB and png
    else:
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE]) #resize of grey images    
    if len(image.shape) > 2:
        image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)  #change of colormap of images alpha chanel delete
    image = np.array(image) / 255

    
    # Reshape input image
    ##YOUR CODE GOES HERE##
    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

        
    # Show image
    ##YOUR CODE GOES HERE##
    # plt.gray()
    # plt.imshow(image)
    # plt.show()

    # Predict the diagnosis
    diag = model.predict(image)
    predicted_class = np.argmax(diag, axis=1)[0]

    # Find the name of the diagnosis  
    ##YOUR CODE GOES HERE##
    predicted_class_name = list(lab.keys())[list(lab.values()).index(predicted_class)]

    return predicted_class_name




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        image = mh.imread(file)
        class_name = diagnosis(image)

        return jsonify({'class': class_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
