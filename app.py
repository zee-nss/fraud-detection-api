from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(tittle = "Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Loading model...")
model = joblib.load('models/fraud_model.pkl')
print("Model Loaded")

class Transaction(BaseModel):
    amount: float
    customer_tenure_days: int
    num_previous_orders: int

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    try:
        features = [[
            transaction.amount,
            transaction.customer_tenure_days,
            transaction.num_previous_orders
        ]]

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0].max())

        if prediction == 1 and probability > 0.8:
            risk_level = "HIGH"
            action = "BLOCK_TRANSACTION"
        elif prediction == 1:
            risk_level = "MEDIUM"
            action = "FLAG_FOR_REVIEW"
        else:
            risk_level = "LOW"
            action = "PROCEED"
        return {
            "Prediction": "fraud" if prediction == 1 else "Legitimate",
            "risk_level": risk_level,
            "confidence": round(probability, 3),
            "action" : action
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

@app.get("/")
def read_root():
    return{"status": "Fraud Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port =8000)