# Plant Disease Classification API Documentation

## Endpoint: `/predict/plant_classifier`

### Overview

This endpoint performs plant classification and disease prediction for uploaded plant images. It uses a multi-stage prediction pipeline that first identifies the plant type and then predicts the specific disease affecting that plant.

### Method

`POST`

### URL

```
/predict/plant_classifier
```

### Content-Type

`multipart/form-data`

### Request Parameters

| Parameter   | Type   | Required | Description                                           |
| ----------- | ------ | -------- | ----------------------------------------------------- |
| `file`      | File   | Yes      | Image file to be analyzed (JPEG, PNG, or WebP format) |
| `plantName` | String | Yes      | Expected plant name for validation/override           |

### Request Example

#### cURL

```bash
curl -X POST "http://localhost:8000/predict/plant_classifier" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/plant_image.jpg" \
  -F "plantName=tomato"
```

#### Python (requests)

```python
import requests

url = "http://localhost:8000/predict/plant_classifier"
files = {
    'file': open('plant_image.jpg', 'rb'),
    'plantName': (None, 'tomato')
}

response = requests.post(url, files=files)
print(response.json())
```

#### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('plantName', 'tomato');

fetch('/predict/plant_classifier', {
  method: 'POST',
  body: formData,
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

### Response Format

#### Success Response (200 OK)

```json
{
  "result": "plant classifier prediction",
  "prediction": "tomato",
  "plantName": "tomato",
  "predictedDisease": "Tomato___Early_blight"
}
```

#### Error Response (400 Bad Request)

```json
{
  "detail": "Only JPEG or PNG images are allowed."
}
```

### Response Fields

| Field              | Type   | Description                                     |
| ------------------ | ------ | ----------------------------------------------- |
| `result`           | String | Status message indicating successful prediction |
| `prediction`       | String | Predicted plant type by the classifier model    |
| `plantName`        | String | The plant name provided in the request          |
| `predictedDisease` | String | The predicted disease affecting the plant       |

### Supported Plant Types

- **chilli** - Chili pepper plants
- **cucumber** - Cucumber plants
- **potato** - Potato plants
- **tomato** - Tomato plants

### Supported Diseases by Plant Type

#### Chilli Diseases

- `healthy` - Healthy plant
- `leaf curl` - Leaf curl disease
- `leaf spot` - Leaf spot disease
- `whitefly` - Whitefly infestation
- `yellowish` - Yellowing disease

#### Potato Diseases

- `Bacteria` - Bacterial infection
- `Fungi` - Fungal infection
- `Healthy` - Healthy plant
- `Nematode` - Nematode infestation
- `Pest` - Pest damage
- `Phytopthora` - Phytophthora infection
- `Virus` - Viral infection

#### Tomato Diseases

- `Tomato___Bacterial_spot` - Bacterial spot
- `Tomato___Early_blight` - Early blight
- `Tomato___healthy` - Healthy plant
- `Tomato___Late_blight` - Late blight
- `Tomato___Leaf_Mold` - Leaf mold
- `Tomato___Septoria_leaf_spot` - Septoria leaf spot
- `Tomato___Spider_mites Two-spotted_spider_mite` - Spider mite infestation
- `Tomato___Target_Spot` - Target spot
- `Tomato___Tomato_mosaic_virus` - Tomato mosaic virus
- `Tomato___Tomato_Yellow_Leaf_Curl_Virus` - Tomato yellow leaf curl virus

#### Cucumber Diseases

- `Anthracnose` - Anthracnose disease
- `Bacterial Wilt` - Bacterial wilt
- `Belly Rot` - Belly rot
- `Downy Mildew` - Downy mildew
- `Fresh Cucumber` - Healthy cucumber
- `Gummy Stem Blight` - Gummy stem blight
- `Pythium Fruit Rot` - Pythium fruit rot

### Technical Details

#### Image Processing

- Images are automatically resized to 256x256 pixels
- RGB color space conversion is applied
- Normalization is performed (pixel values divided by 255.0)

#### Model Architecture

- **Plant Classifier**: Global classifier model (`plant_classifier.h5`)
- **Disease Models**:
  - Chilli model (`chilli_model.h5`)
  - Potato model (`potato_model.h5`)
  - Cucumber model (`cucumber_model.h5`)
  - Tomato model (`tomato_model.h5`)

#### Prediction Pipeline

1. **Image Preprocessing**: Resize and normalize uploaded image
2. **Plant Classification**: Use global classifier to identify plant type
3. **Plant Validation**: Compare predicted plant with provided plantName
4. **Disease Prediction**: Use plant-specific model to predict disease
5. **Response Generation**: Return comprehensive prediction results

### Error Handling

| HTTP Status | Error Condition     | Description                                   |
| ----------- | ------------------- | --------------------------------------------- |
| 400         | Invalid file format | Only JPEG, PNG, and WebP images are supported |
| 400         | Invalid image data  | Corrupted or unreadable image file            |

### Rate Limiting

Currently, no rate limiting is implemented.

### Authentication

No authentication is required for this endpoint.

### Dependencies

- FastAPI
- TensorFlow
- PIL (Python Imaging Library)
- NumPy

### Example Use Cases

1. **Farm Management**: Farmers can upload plant images to get instant disease diagnosis
2. **Agricultural Research**: Researchers can use the API for plant health monitoring
3. **Educational Purposes**: Students can learn about plant diseases through image analysis
4. **Crop Monitoring**: Automated systems can integrate this API for continuous crop health assessment

### Notes

- The API uses a proxy plant label system where the provided `plantName` takes precedence if it's in the supported plant classes
- If the provided `plantName` is not recognized, the system falls back to the predicted plant type
- All models are pre-loaded at server startup for optimal performance
