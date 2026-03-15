# Brain Tumor Classification with Human-Readable MRI Explanations Using Deep Learning

Magnetic Resonance Imaging (MRI) scans are commonly used by doctors to examine the brain and identify abnormal growths or tumors. The goal of this project is to build an AI system that can automatically analyze brain MRI images to classify MRI scans into four categories: glioma, meningioma, pituitary tumor, or normal, and then generate human-readable explanations describing the detected condition.
We also developed a Flask API and a user interface that allow MRI images to be uploaded and analyzed interactively while giving flexibility for integration into existing medical diagnosis systems or insurance adjudication workflows to provide AI support in clinical review and decision-making.


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
python -m pip install -r requirements.txt
cd backend
python app.py
```

## Frontend

```
cd /Users/mathuria/Desktop/MSDS462_Capstone_Project/frontend
npm install
npm run dev
```

