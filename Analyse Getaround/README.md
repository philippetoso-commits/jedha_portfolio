# GetAround Analysis & Deployment

This repository contains the deliverables for the **GetAround** pricing and delay analysis project (Jedha Bootcamp Bloc 5).

##  1. Streamlit Dashboard (Delay Analysis)
The dashboard helps the Product Manager visualize the trade-off between reducing friction (problems solved) and protecting revenue (rentals affected) when implementing a minimum delay between rentals.

 **[Live Dashboard](https://philippetos-getaround.hf.space)**  
 *[Hugging Face Space](https://huggingface.co/spaces/philippetos/GetAround)*

##  2. FastAPI (Pricing Prediction ML)
A deployed Machine Learning API using FastAPI that predicts the optimal rental price per day based on a car's characteristics. 

 **[Live API (Swagger UI)](https://philippetos-getaroundapi.hf.space/docs)**  
 *[Hugging Face Space](https://huggingface.co/spaces/philippetos/GetAroundAPI)*

**Test the API endpoints from your terminal:**

**1. Check API Health:**
```bash
curl -X 'GET' 'https://philippetos-getaroundapi.hf.space/health' -H 'accept: application/json'
```

**2. Make a Price Prediction (Single Car):**
```bash
curl -i -H "Content-Type: application/json" -X POST -d '{
  "model_key": "Renault",
  "mileage": 50000,
  "engine_power": 120,
  "fuel": "diesel",
  "paint_color": "grey",
  "car_type": "sedan",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": true,
  "automatic_car": false,
  "has_getaround_connect": false,
  "has_speed_regulator": true,
  "winter_tires": true
}' https://philippetos-getaroundapi.hf.space/predict
```

**3. Get Dataset Statistics:**
```bash
curl -X 'GET' 'https://philippetos-getaroundapi.hf.space/cars/stats' -H 'accept: application/json'
```

##  3. Data Science Notebooks
The complete exploratory data analysis (EDA), data cleaning, feature engineering, and Machine Learning pipeline (with MLflow tracking) are available in the `notebooks/` directory.
- `GetAround_EDA_ML_EN.ipynb` (English)
- `GetAround_EDA_ML_FR.ipynb` (French translation)

##  4. Presentation & Documentation
- `Presentation_GetAround.md`: Slides summarizing the business problem, methods, results, and deployment.
- `discours.md`: The 10-minute speech script to present the project.
- `FAQ.md`: Anticipated questions and detailed answers for the Q&A session.
