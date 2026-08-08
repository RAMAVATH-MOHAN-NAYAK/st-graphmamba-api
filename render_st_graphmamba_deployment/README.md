# ST-GraphMamba Render API

FastAPI backend for the traffic forecasting frontend.

## Render

Create a **Web Service**:

- Runtime: Python
- Plan: Free
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Set the environment variable:

`MODEL_REPO_ID=YOUR_USERNAME/YOUR_MODEL_REPO`

If the Model Repo is private, set the secret:

`HF_TOKEN=hf_...`

The API downloads the trained artifacts from Hugging Face.

## Endpoints

`GET /health`

`GET /predict/latest`

`POST /predict`

`POST /predict/file`

## Important

The included `model.py` and `preprocess.py` must match the architecture and
preprocessing used to train the checkpoint. Replace them with your exact
training versions if your Model Repo contains different versions.
