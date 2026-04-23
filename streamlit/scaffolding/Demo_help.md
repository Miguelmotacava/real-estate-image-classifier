# Scaffolding: Run API + Streamlit

This folder includes:
- `fastapi_backend.py` (backend API)
- `app.py` (Streamlit frontend)

## 1) Install dependencies

From the repository root (`DeepLearning_Lab`):

```bash
pip install fastapi uvicorn
```

## 2) Run the FastAPI backend

Open Terminal 1 and run:

```bash
cd streamlit/scaffolding
python fastapi_backend.py
```

The API will be available at:
- `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

## 3) Run the Streamlit app

Open Terminal 2 (same virtual environment) and run:

```bash
cd streamlit/scaffolding
streamlit run app.py
```

Streamlit will open in your browser (usually `http://localhost:8501`).

## 4) Test flow

1. Keep API running on port `8000`.
2. Open Streamlit on port `8501`.
3. Upload an image and click **Predict**.
4. Streamlit sends the file to `POST /predict` and shows the mock prediction.
