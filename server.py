from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import shutil
from io import BytesIO
from typing import Annotated
from tensorflow.keras.preprocessing import image
from utils import preprocess_image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_classifier_model = tf.keras.models.load_model("models/plant_classifier.h5")
chilli_model = tf.keras.models.load_model("models/chilli_model.h5")
potato_model = tf.keras.models.load_model("models/potato_model.h5")
cucumber_model = tf.keras.models.load_model("models/cucumber_model.h5")
tomato_model = tf.keras.models.load_model("models/tomato_model.h5")

plant_classes = ['chilli', 'cucumber', 'potato', 'tomato']
chilli_diseases = ['healthy','leaf curl','leaf spot','whitefly','yellowish']
potato_diseases =['Bacteria','Fungi','Healthy','Nematode','Pest','Phytopthora','Virus']
tomato_diseases = ['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']
cucumber_diseases = ['Anthracnose','Bacterial Wilt','Belly Rot','Downy Mildew','Fresh Cucumber','Gummy Stem Blight','Pythium Fruit Rot']

                   


async def process_generate_img(file):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((256, 256)) 

    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array


async def prefict_plant_using_model(img_array):
    # processed_image = preprocess_image(contents)
    predictions = global_classifier_model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=-1)
    predicted_class_name = plant_classes[predicted_class_idx[0]]

    return predicted_class_name



def prefict_disease_using_model(model, classes, img_array):
    # img = image.load_img(file,target_size=(256,256))
    # img_array = image.img_to_array(img)
    # img_array=np.expand_dims(img_array,axis=0)
    # img_array = img_array/255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)
    print(f"Predicted class id.: {predicted_class}")
    print(f"Predicted class: {classes[predicted_class[0]]}")
    return classes[predicted_class[0]]

# Helper function to preprocess image

def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file)).convert('RGB')
    image = image.resize((224, 224))  # Assuming models expect 224x224
    return np.array(image)

@app.get("/check")
def check():
    return JSONResponse(content={"status": "API is working"})

@app.post("/upload-image/")
async def upload_image(file: Annotated[UploadFile, File(...)]):
    # Validate MIME type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are allowed.")

    # Read image bytes into memory
    contents = await file.read()
    
    try:
        # Use PIL to verify it's a valid image
        image = Image.open(BytesIO(contents))
        width, height = image.size
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    return JSONResponse(content={
        "filename": file.filename,
        "content_type": file.content_type,
        "dimensions": {"width": width, "height": height}
    })


# @app.post("/predict/potato")
def predict_potato(img_array):
    predicted_disease = prefict_disease_using_model(potato_model, potato_diseases, img_array)
    # return {"disease prediction result": predicted_disease}
    return predicted_disease

# @app.post("/predict/tomato")
def predict_tomato(img_array):
    predicted_disease = prefict_disease_using_model(tomato_model, tomato_diseases, img_array)
    # return {"disease prediction result": predicted_disease}
    return predicted_disease

# @app.post("/predict/chilli")
def predict_chilli(img_array):
    predicted_disease = prefict_disease_using_model(chilli_model, chilli_diseases, img_array)
    # return {"disease prediction result": predicted_disease}
    return predicted_disease

# @app.post("/predict/cucumber")
def predict_cucumber(img_array):
    predicted_disease = prefict_disease_using_model(cucumber_model, cucumber_diseases, img_array)
    # return {"disease prediction result": predicted_disease}
    return predicted_disease


@app.post("/predict/plant_classifier")
async def predict_plant_classifier(
    file: UploadFile = File(...),
    plantName: str = Form(...)
):
    
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are allowed.")
    
    # # image = Image.open(file)
    # contents = await file.read()
    # img = Image.open(BytesIO(contents)).convert("RGB")
    # img = img.resize((256, 256))  # Resize to match training
    # # Convert to array and preprocess
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = img_array / 255.0

    # # processed_image = preprocess_image(contents)
    # predictions = global_classifier_model.predict(img_array)
    # predicted_class_idx = np.argmax(predictions, axis=-1)
    # predicted_class_name = plant_classes[predicted_class_idx[0]]

    # print(predicted_class_name)
    # return {"result": "plant classifier prediction (dummy)","prediction": predicted_class_name}

    # process the image
    img_array = await process_generate_img(file)

    # predict the plant
    predicted_class_name = await prefict_plant_using_model(img_array)
    proxy_plant_label = plantName if plantName in plant_classes else predicted_class_name
    disease = ''
    if proxy_plant_label == 'chilli':
        disease = predict_chilli(img_array)
    if proxy_plant_label == 'potato':
        disease = predict_potato(img_array)
    if proxy_plant_label == 'tomato':
        disease = predict_tomato(img_array)
    if proxy_plant_label == 'cucumber':
        disease = predict_cucumber(img_array)



    return {
        "result": "plant classifier prediction",
        "prediction": predicted_class_name,
        "plantName": plantName,
        "predictedDisease": disease
    }



# To run: uvicorn server:app --reload


