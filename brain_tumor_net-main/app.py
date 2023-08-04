from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

import torch
import torch.nn as nn


class BrainTumorNet(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_transform(image):
    # Pastikan gambar memiliki 3 saluran warna (RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Ubah ukuran gambar menjadi 224x224
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    transform = transforms.Compose([
            transforms.Resize(256),  # Resize gambar menjadi ukuran 256x256
            transforms.CenterCrop(224),  # Crop gambar menjadi ukuran 224x224 dari tengah
            transforms.ToTensor(),  # Konversi gambar menjadi tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisasi gambar
        ])
    return transform(image)


# classes = ['NORMAL', 'PNEUMONIA']  # Ganti dengan nama kelas Anda
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/')
def index():
    return render_template('index.html', appName="Pneumonia Detection")

@app.route('/predictAPI', methods=["POST"])
def api():
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        
        image = request.files.get('fileup')

        if image and allowed_file(image.filename):
            image_arr = Image.open(image)
            image_arr = image_transform(image_arr).unsqueeze(0).to(device)
            print("Model predicting ...")
            
            with torch.no_grad():
                output = model(image_arr)
            
            print("Model predicted")
            probabilities = torch.softmax(output, dim=1)[0]
            pneumonia_probability = probabilities[1].item()
            pneumonia_percentage = pneumonia_probability * 100

            ind = torch.argmax(probabilities).item()
            prediction = classes[ind]

            print(prediction)
            return jsonify({'prediction': prediction})
        else:
            return "Invalid file format. Please upload a valid image file (JPG, PNG, or JPEG)."
    except Exception as e:
        return jsonify({'Error': str(e)})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            image = request.files['fileup']

            if image and allowed_file(image.filename):
                image_arr = Image.open(image)
                image_arr = image_transform(image_arr).unsqueeze(0).to(device)
                print("Predicting ...")
                with torch.no_grad():
                    output = model(image_arr)
                print("Predicted ...")
                probabilities = torch.softmax(output, dim=1)[0]
                pneumonia_probability = probabilities[1].item()
                pneumonia_percentage = pneumonia_probability * 100 

                ind = torch.argmax(probabilities).item()
                prediction = classes[ind]

                return render_template('index.html', prediction=prediction, probability=pneumonia_percentage, image='/style/IMG/', appName="Pneumonia Detection")
            else:
                return render_template('index.html', appName="Pneumonia Detection")
        except:
            return render_template('index.html', appName="Pneumonia Detection")
    else:
        return render_template('index.html', appName="Pneumonia Detection")

if __name__ == '__main__':
    checkpoint_path = 'checkpoint.pt'
    config = {
        'num_classes' : 4
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = BrainTumorNet(config['num_classes'])
    checkpoint = torch.load(checkpoint_path,  map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    app.run(debug=True)
