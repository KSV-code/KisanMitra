from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text

from backend.database import engine
from ml.test_crop_prediction import predict_crops
from recommendation.scheme_recommender import recommend_all_schemes

app = FastAPI(title="KisanMitra API")


class Farmer(BaseModel):
    name: str
    phone: str | None = None
    email: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None

class Farm(BaseModel):
    farmer_id: int
    land_area: float | None = None
    area_unit: str | None = None
    location: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    soil_type: str | None = None
    irrigation_type: str | None = None
    farming_type: str | None = None

class Crop(BaseModel):
    farm_id: int
    crop_name: str
    variety: str | None = None
    season: str | None = None
    sowing_date: date | None = None
    harvest_date: date | None = None
    previous_yield: float | None = None
    yield_unit: str | None = None

class CropPredictionInput(BaseModel):
    N: float
    Fe: float
    Mn: float
    S: float
    Mg: float
    C: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class SchemeRecommendationInput(BaseModel):
    state: str | None = None
    owns_cultivable_land: bool
    income_tax_payer: bool
    government_employee: bool
    monthly_pension: float
    practicing_professional: bool
    nri_farmer: bool
    grows_notified_crop: bool
    in_notified_area: bool
    has_insurable_interest: bool
    has_required_documents: bool
    is_farmer: bool
    has_cultivation_activity: bool
    requires_soil_testing: bool
    practices_rainfed_agriculture: bool

@app.get("/")
def root():
    return {"message": "KisanMitra API is running"}


@app.post("/farmers")
def create_farmer(farmer: Farmer):

    with engine.begin() as connection:
        result = connection.execute(
            text("""
                INSERT INTO farmers
                (name, phone, email, date_of_birth, gender)
                VALUES
                (:name, :phone, :email, :date_of_birth, :gender)
                RETURNING farmer_id
            """),
            {
                "name": farmer.name,
                "phone": farmer.phone,
                "email": farmer.email,
                "date_of_birth": farmer.date_of_birth,
                "gender": farmer.gender,
            }
        )

        farmer_id = result.scalar_one()

    return {
        "message": "Farmer created successfully",
        "farmer_id": farmer_id,
        "farmer": farmer
    }

@app.get("/farmers")
def get_farmers():

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM farmers
                ORDER BY farmer_id
            """)
        )

        farmers = [dict(row._mapping) for row in result]

    return farmers

@app.get("/farmers/{farmer_id}")
def get_farmer(farmer_id: int):

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM farmers
                WHERE farmer_id = :farmer_id
            """),
            {"farmer_id": farmer_id}
        )

        farmer = result.fetchone()

    if farmer is None:
        return {"message": "Farmer not found"}

    return dict(farmer._mapping)

@app.get("/farmers/{farmer_id}/scheme-profile")
def get_scheme_profile(farmer_id: int):

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM farmer_scheme_profiles
                WHERE farmer_id = :farmer_id
            """),
            {"farmer_id": farmer_id}
        )

        profile = result.fetchone()

    if profile is None:
        return {"message": "Scheme profile not found"}

    return dict(profile._mapping)

@app.post("/farms")
def create_farm(farm: Farm):

    with engine.begin() as connection:
        result = connection.execute(
            text("""
                INSERT INTO farms
                (
                    farmer_id,
                    land_area,
                    area_unit,
                    location,
                    latitude,
                    longitude,
                    soil_type,
                    irrigation_type,
                    farming_type
                )
                VALUES
                (
                    :farmer_id,
                    :land_area,
                    :area_unit,
                    :location,
                    :latitude,
                    :longitude,
                    :soil_type,
                    :irrigation_type,
                    :farming_type
                )
                RETURNING farm_id
            """),
            {
                "farmer_id": farm.farmer_id,
                "land_area": farm.land_area,
                "area_unit": farm.area_unit,
                "location": farm.location,
                "latitude": farm.latitude,
                "longitude": farm.longitude,
                "soil_type": farm.soil_type,
                "irrigation_type": farm.irrigation_type,
                "farming_type": farm.farming_type,
            }
        )

        farm_id = result.scalar_one()


    return {
        "message": "Farm created successfully",
        "farm_id": farm_id,
        "farm": farm
    }

@app.get("/farms")
def get_farms():

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM farms
                ORDER BY farm_id
            """)
        )

        farms = [dict(row._mapping) for row in result]

    return farms

@app.post("/crops")
def create_crop(crop: Crop):

    with engine.begin() as connection:
        result = connection.execute(
            text("""
                INSERT INTO crops
                (
                    farm_id,
                    crop_name,
                    variety,
                    season,
                    sowing_date,
                    harvest_date,
                    previous_yield,
                    yield_unit
                )
                VALUES
                (
                    :farm_id,
                    :crop_name,
                    :variety,
                    :season,
                    :sowing_date,
                    :harvest_date,
                    :previous_yield,
                    :yield_unit
                )
                RETURNING crop_id
            """),
            {
                "farm_id": crop.farm_id,
                "crop_name": crop.crop_name,
                "variety": crop.variety,
                "season": crop.season,
                "sowing_date": crop.sowing_date,
                "harvest_date": crop.harvest_date,
                "previous_yield": crop.previous_yield,
                "yield_unit": crop.yield_unit,
            }
        )

        crop_id = result.scalar_one()

    return {
        "message": "Crop created successfully",
        "crop_id": crop_id,
        "crop": crop
    }

@app.get("/crops")
def get_crops():

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM crops
                ORDER BY crop_id
            """)
        )

        crops = [dict(row._mapping) for row in result]

    return crops

@app.get("/government-schemes")
def get_government_schemes():

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM government_schemes
                ORDER BY scheme_id
            """)
        )

        schemes = [dict(row._mapping) for row in result]

    return schemes

@app.post("/predict-crops")
def predict_crop_recommendation(data: CropPredictionInput):

    farmer_data = data.model_dump()

    recommendations = predict_crops(farmer_data)

    return {
        "recommendations": recommendations
    }

@app.post("/recommend-schemes")
def recommend_government_schemes(data: SchemeRecommendationInput):

    farmer_data = data.model_dump()

    recommendations = recommend_all_schemes(farmer_data)

    return {
        "recommendations": recommendations
    }

@app.get("/farmers/{farmer_id}/recommend-schemes")
def recommend_schemes_for_farmer(farmer_id: int):

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM farmer_scheme_profiles
                WHERE farmer_id = :farmer_id
            """),
            {"farmer_id": farmer_id}
        )

        profile = result.fetchone()

    if profile is None:
        return {"message": "Scheme profile not found"}

    farmer_data = dict(profile._mapping)

    farmer_data.pop("profile_id", None)
    farmer_data.pop("farmer_id", None)

    recommendations = recommend_all_schemes(farmer_data)

    return {
        "farmer_id": farmer_id,
        "recommendations": recommendations
    }