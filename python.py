from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

app = Flask(__name__)
# Specify the desired image size used during training
IMAGE_SIZE = (50, 50)
label_to_character = {
    1: 'అ',  # Replace with the corresponding Telugu character for label 1
    2: 'ఆ',  # Replace with the corresponding Telugu character for label 2
    3: 'అః',
    4: 'ఐ',
    5: 'అం',
    6: 'ఔ',
    7: 'ఇ',
    8: 'ఈ',
    9: 'ఎ',
    10: 'ఏ',
    11: 'ఒ',
    12: 'ఓ',
    13: 'ఋ',
    14: 'ౠ',
    15: 'ఉ',
    16: 'ఊ',
    17:'ణ',
    18:'బ',
    19:'భ',
    20:'చ',
    21:'ఛ',
    22:'డ',
    23:'ఢ',
    24:'ద',
    25:'ధ',
    26:'గ',
    27:'ఘ',
    28:'హ',
    29:'జ',
    30:'ఝ',
    31:'ఙ',
    32:'క',
    33:'ఖ',
    34:'క్ష',
    35:'ల',
    36:'ళ',
    37:'మ',
    38:'న',
    39:'ప',
    40:'ఫ',
    41:'ర',
    42:'ఱ',
    43:'శ',
    44:'స',
    45:'ష',
    46:'ట',
    47:'ఠ',
    48:'త',
    49:'థ',
    50:'వ',
    51:'య',
}

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    class_labels = [i for i in range(1,52)]  # List of class labels
    if request.method == 'POST':
        file = request.files['image']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # Convert the image to RGB if it has an alpha channel
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Resize the image to match the desired image size
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
             
        model = load_model('model1.h5')
        pred_y = model.predict(img)
        max_one = np.argmax(pred_y)
        predicted_label = class_labels[max_one]
        predicted_character = label_to_character.get(predicted_label, 'Unknown')

        return predicted_character

if __name__ == "__main__":
    app.run(debug=True)
