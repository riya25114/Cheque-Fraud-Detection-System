import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template

# ============================
# Siamese Network (Colab version)
# ============================
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)  # output similarity score
        )
    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# ============================
# Flask app
# ============================
app = Flask(__name__)

# Image preprocessing (same as Colab)
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
MODEL_PATH = "siamese_signature_model.pth"   # make sure this file exists in same folder
model = SiameseNetwork()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Save uploaded images temporarily
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ============================
# Routes
# ============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "image1" not in request.files or "image2" not in request.files:
            result = "Please upload two images."
        else:
            img1 = request.files["image1"]
            img2 = request.files["image2"]

            # Save temporarily
            img1_path = os.path.join(app.config["UPLOAD_FOLDER"], img1.filename)
            img2_path = os.path.join(app.config["UPLOAD_FOLDER"], img2.filename)
            img1.save(img1_path)
            img2.save(img2_path)

            # Preprocess
            def preprocess(path):
                image = Image.open(path).convert("L")  # grayscale
                return transform(image).unsqueeze(0)   # add batch dim

            img1_tensor = preprocess(img1_path)
            img2_tensor = preprocess(img2_path)

            with torch.no_grad():
                output1, output2 = model(img1_tensor, img2_tensor)
                distance = F.pairwise_distance(output1, output2)

            # Threshold (you can adjust, try 1.0 or 1.5 depending on training)
            if distance.item() < 1.0:
                result = "✅ Signatures Match (Genuine)"
            else:
                result = "❌ Signatures Do Not Match (Forged)"

    return render_template("index.html", result=result)

# ============================
# Run server
# ============================
if __name__ == "__main__":
    app.run(debug=True)