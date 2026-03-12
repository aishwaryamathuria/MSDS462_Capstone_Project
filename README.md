# Brain Tumor Classification with Human-Readable MRI Explanations Using Deep Learning

## Project Structure

- `backend/`: Flask API service to serves model inference and provide explanation endpoints.
- `frontend/`: React + Vite web app for uploading MRI images and viewing predictions.
- `brain_tumor_densenet121_training.ipynb`: Training and evaluation code for DenseNet121 tumor classification model.
- `data_analysis/data_analysis.ipynb`: Exploratory analysis on the MRI dataset and labels.
- `model_exploration/VLM_explanation.ipynb`: Tests VLM prompts and explanation quality for model outputs.

## Backend

```
cd /Users/mathuria/Desktop/MSDS462_Capstone_Project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --force-reinstall numpy torch torchvision
pip install flask pillow transformers accelerate sentencepiece safetensors
cd backend
python app.py
```

## Frontend

```
cd /Users/mathuria/Desktop/MSDS462_Capstone_Project/frontend
npm install
npm run dev
```

