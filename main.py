import os
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

app = FastAPI(title="Life Expectancy Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ---------- #
class PredictRequest(BaseModel):
    full_name: Optional[str] = Field(None, description="User's name")
    birth_date: str = Field(..., description="ISO date string YYYY-MM-DD")
    gender: Optional[str] = Field("unspecified", description="male | female | unspecified")
    is_smoker: Optional[bool] = False
    height_cm: Optional[float] = Field(None, ge=100, le=250)
    weight_kg: Optional[float] = Field(None, ge=30, le=300)
    weekly_exercise_mins: Optional[int] = Field(0, ge=0, le=2000)
    stress_level: Optional[int] = Field(3, ge=1, le=5, description="1-5")
    country: Optional[str] = Field("global")

    @validator("birth_date")
    def validate_birth_date(cls, v: str):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("birth_date must be YYYY-MM-DD")


class PredictResponse(BaseModel):
    name: Optional[str]
    birth_date: date
    current_age_years: float
    predicted_lifespan_years: float
    predicted_death_date: date
    remaining_years: float
    confidence: int
    factors: dict


# ---------- Helpers ---------- #
BASE_LIFE_EXPECTANCY = {
    "global": {"male": 71.0, "female": 75.6, "unspecified": 73.3},
    "usa": {"male": 73.2, "female": 79.1, "unspecified": 76.2},
    "uk": {"male": 79.0, "female": 82.9, "unspecified": 81.0},
    "india": {"male": 67.5, "female": 70.7, "unspecified": 69.1},
    "japan": {"male": 81.5, "female": 87.6, "unspecified": 84.5},
    "nigeria": {"male": 53.5, "female": 55.2, "unspecified": 54.4},
}


def parse_country(key: Optional[str]) -> str:
    if not key:
        return "global"
    k = key.strip().lower()
    return k if k in BASE_LIFE_EXPECTANCY else "global"


def calc_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if not height_cm or not weight_kg:
        return None
    m = height_cm / 100.0
    if m <= 0:
        return None
    return round(weight_kg / (m * m), 1)


def estimate_lifespan_years(req: PredictRequest) -> tuple[float, dict, int]:
    # Start with base by country and gender
    country_key = parse_country(req.country)
    base = BASE_LIFE_EXPECTANCY[country_key].get(req.gender or "unspecified", BASE_LIFE_EXPECTANCY[country_key]["unspecified"])

    adjustments: dict[str, float] = {}
    confidence = 70  # start medium

    # Smoking impact
    if req.is_smoker:
        adjustments["smoking"] = -7.0
        confidence -= 5
    else:
        adjustments["non_smoker_bonus"] = +1.5
        confidence += 2

    # Exercise: up to +2.5 years for 150-300 mins/week
    ex = req.weekly_exercise_mins or 0
    if ex >= 300:
        adjustments["exercise"] = +2.5
        confidence += 2
    elif ex >= 150:
        adjustments["exercise"] = +1.5
    elif ex > 0:
        adjustments["exercise"] = +0.5

    # Stress: scale -0.5 to -3.0 years
    stress = req.stress_level or 3
    stress_penalty = {1: -0.5, 2: -1.0, 3: -1.5, 4: -2.2, 5: -3.0}[stress]
    adjustments["stress"] = stress_penalty

    # BMI impact
    bmi = calc_bmi(req.height_cm, req.weight_kg)
    if bmi is not None:
        if bmi < 18.5:
            adjustments["bmi"] = -1.5
        elif bmi < 25:
            adjustments["bmi"] = +1.0
        elif bmi < 30:
            adjustments["bmi"] = -1.0
        elif bmi < 35:
            adjustments["bmi"] = -3.0
        else:
            adjustments["bmi"] = -6.0
        confidence += 3

    # Gender minor tweak if unspecified
    if (req.gender or "unspecified").lower() == "unspecified":
        adjustments["gender_unspecified"] = -0.3

    lifespan = base + sum(adjustments.values())
    lifespan = max(40.0, min(100.0, lifespan))
    confidence = max(50, min(90, confidence))

    return lifespan, adjustments, confidence


# ---------- Routes ---------- #
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.post("/api/predict", response_model=PredictResponse)
def predict_life(req: PredictRequest):
    try:
        dob = datetime.strptime(req.birth_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid birth_date format. Use YYYY-MM-DD")

    today = date.today()
    if dob > today:
        raise HTTPException(status_code=400, detail="birth_date cannot be in the future")

    age_days = (today - dob).days
    age_years = round(age_days / 365.2425, 2)

    lifespan_years, factors, confidence = estimate_lifespan_years(req)

    remaining_years = max(0.0, lifespan_years - age_years)
    predicted_death_date = today + timedelta(days=int(remaining_years * 365.2425))

    return PredictResponse(
        name=req.full_name,
        birth_date=dob,
        current_age_years=round(age_years, 2),
        predicted_lifespan_years=round(lifespan_years, 2),
        predicted_death_date=predicted_death_date,
        remaining_years=round(remaining_years, 2),
        confidence=confidence,
        factors={k: round(v, 2) for k, v in factors.items()},
    )


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
