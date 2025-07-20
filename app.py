import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './uploads')
app.config['PORT'] = int(os.getenv('PORT', 8080))
openai.api_key = os.getenv('OPENAI_API_KEY')

resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

tumor_detection_model = tf.keras.models.load_model(os.getenv('MODEL_PATH_DETECTION', './models/Detection_Model.keras'))
tumor_classification_model = tf.keras.models.load_model(
    os.getenv('MODEL_PATH_CLASSIFICATION', './models/tumor_classification_model.keras'))

class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    image = preprocess_input(image.astype(np.float32))
    image = np.expand_dims(image, axis=0)
    features = resnet_model.predict(image)
    return features.flatten()


dataset_path = os.getenv('DATASET_PATH', './validation')
dataset_features = [extract_features(os.path.join(dataset_path, filename)) for filename in os.listdir(dataset_path)]
dataset_features = np.array(dataset_features)


def predict_tumor_and_type(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    detection_prediction = tumor_detection_model.predict(img_array)
    if detection_prediction <= 0.5:
        classification_prediction = tumor_classification_model.predict(img_array)
        predicted_class = np.argmax(classification_prediction)
        predicted_tumor_type = class_labels[predicted_class]
        tumor_info, treatment_plan = get_tumor_info_and_solution(predicted_tumor_type)
        return {"tumor_detected": True, "tumor_type": predicted_tumor_type, "tumor_info": tumor_info,
                "treatment_plan": treatment_plan}

    return {"tumor_detected": False, "tumor_type": "notumor", "tumor_info": "No tumor detected.",
            "treatment_plan": "No treatment required."}


def get_tumor_info_and_solution(tumor_type):
    if tumor_type == "notumor":
        return "No tumor detected.", "No treatment required."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a medical expert providing concise tumor information and treatments."},
                {"role": "user",
                 "content": f"Provide a 2-3 line description of {tumor_type} tumor and a 2-3 line solution for its treatment."}
            ]
        )
        content = response['choices'][0]['message']['content'].strip().split("\n")
        tumor_info = " ".join(content[:2]).strip()
        treatment_plan = " ".join(content[2:4]).strip()
        return tumor_info, treatment_plan
    except Exception as e:
        print(f"Error fetching tumor info from OpenAI: {e}")
        return "Information unavailable.", "Consult a doctor."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    prediction_result = predict_tumor_and_type(filepath)

    return jsonify({
        "image_url": f"/uploads/{filename}",
        "tumor_detected": prediction_result["tumor_detected"],
        "tumor_type": prediction_result["tumor_type"],
        "tumor_info": prediction_result["tumor_info"],
        "treatment_plan": prediction_result["treatment_plan"]
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/get_explanation', methods=['POST'])
def get_explanation():
    tumor_type = request.json.get("tumor_type")
    if tumor_type and tumor_type in class_labels:
        try:
            explanation, treatment = get_tumor_info_and_solution(tumor_type)
            return jsonify({"explanation": explanation, "treatment_plan": treatment})
        except Exception as e:
            return jsonify({"error": f"Error fetching explanation: {e}"}), 500
    return jsonify({"error": "Invalid or missing tumor_type in request."}), 400


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=app.config['PORT'])

