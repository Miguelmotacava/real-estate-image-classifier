from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI(title="Image Classifier API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the image file
    # contents = await file.read() 
    
    # 2. This is where their ML Model logic goes
    # result = my_model.predict(contents)
    
    # Mock Response for now:
    return {
        "filename": file.filename,
        "label": "Golden Retriever",
        "confidence": 0.98,
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)