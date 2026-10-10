from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from pathlib import Path
from uuid import uuid4

from backend.database import engine
from ml.test_crop_prediction import predict_crops
from recommendation.scheme_recommender import recommend_all_schemes
from passlib.context import CryptContext
from jose import jwt

from fastapi import Depends, HTTPException, Form, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="KisanMitra API")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing from .env")

ALGORITHM = "HS256"
security = HTTPBearer()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    return payload

def require_admin(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return current_user

@app.get("/admin-test")
def admin_test(current_user: dict = Depends(require_admin)):
    return {
        "message": "Admin access successful",
        "user_id": current_user["user_id"],
        "role": current_user["role"]
    }

@app.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "message": "Authentication successful",
        "user_id": current_user["user_id"],
        "role": current_user["role"]
    }

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

class FarmerDocument(BaseModel):
    farmer_id: int
    document_type: str
    document_name: str
    document_path: str

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str


@app.post("/register")
def register_user(data: UserRegister):
    password_hash = pwd_context.hash(data.password)

    try:
        with engine.begin() as connection:
            existing_user = connection.execute(
                text("""
                    SELECT user_id
                    FROM users
                    WHERE email = :email
                """),
                {"email": data.email}
            ).fetchone()

            if existing_user:
                raise HTTPException(
                    status_code=409,
                    detail="Email already registered"
                )

            farmer_result = connection.execute(
                text("""
                    INSERT INTO farmers (name, email)
                    VALUES (:name, :email)
                    RETURNING farmer_id
                """),
                {
                    "name": data.name,
                    "email": data.email
                }
            )

            farmer_id = farmer_result.scalar_one()

            user_result = connection.execute(
                text("""
                    INSERT INTO users
                    (name, email, password_hash, role, farmer_id)
                    VALUES
                    (:name, :email, :password_hash, 'farmer', :farmer_id)
                    RETURNING user_id, name, email, role, farmer_id
                """),
                {
                    "name": data.name,
                    "email": data.email,
                    "password_hash": password_hash,
                    "farmer_id": farmer_id
                }
            )

            user = user_result.fetchone()

    except IntegrityError as exc:
        if "users_email_unique" in str(exc.orig):
            raise HTTPException(
                status_code=409,
                detail="Email already registered"
            ) from exc
        raise

    return {
        "message": "Registration successful",
        "user": dict(user._mapping)
    }


@app.post("/login")
def login_user(data: UserLogin):

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT user_id, name, email, password_hash, role, farmer_id
                FROM users
                WHERE email = :email
            """),
            {"email": data.email}
        )

        user = result.fetchone()

    if user is None:
        return {"message": "Invalid email or password"}

    if not pwd_context.verify(data.password, user.password_hash):
        return {"message": "Invalid email or password"}

    token_data = {
        "user_id": user.user_id,
        "role": user.role,
        "farmer_id": user.farmer_id
    }

    access_token = jwt.encode(
        token_data,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
            "farmer_id": user.farmer_id
        }
    }

@app.get("/")
def root():
    return {"message": "KisanMitra API is running"}



@app.post("/farmers")
def create_farmer(
    farmer: Farmer,
    current_user: dict = Depends(require_admin)
):
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
def get_farmers(
    current_user: dict = Depends(require_admin)
):
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
def get_farmer(
    farmer_id: int,
    current_user: dict = Depends(get_current_user)
):
    if (
        current_user.get("role") != "admin"
        and current_user.get("farmer_id") != farmer_id
    ):
        raise HTTPException(
            status_code=403,
            detail="You can only access your own farmer record"
        )

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
def get_scheme_profile(
    farmer_id: int,
    current_user: dict = Depends(get_current_user)
):
    if (current_user.get("role") != "admin" and current_user.get("farmer_id") != farmer_id):
        raise HTTPException(
            status_code=403,
            detail="You can only access your own scheme profile"
        )

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
def create_farm(
    farm: Farm,
    current_user: dict = Depends(get_current_user)
):
    if (
        current_user.get("role") != "admin"
        and current_user.get("farmer_id") != farm.farmer_id
    ):
        raise HTTPException(
            status_code=403,
            detail="You can only create farms for yourself"
        )

    with engine.begin() as connection:
        result = connection.execute(
            text("""
                INSERT INTO farms
                (
                    farmer_id, land_area, area_unit, location,
                    latitude, longitude, soil_type,
                    irrigation_type, farming_type
                )
                VALUES
                (
                    :farmer_id, :land_area, :area_unit, :location,
                    :latitude, :longitude, :soil_type,
                    :irrigation_type, :farming_type
                )
                RETURNING farm_id
            """),
            farm.model_dump()
        )
        farm_id = result.scalar_one()

    return {
        "message": "Farm created successfully",
        "farm_id": farm_id,
        "farm": farm
    }


@app.get("/farms")
def get_farms(
    current_user: dict = Depends(get_current_user)
):
    with engine.connect() as connection:
        if current_user.get("role") == "admin":
            result = connection.execute(
                text("SELECT * FROM farms ORDER BY farm_id")
            )
        else:
            result = connection.execute(
                text("""
                    SELECT * FROM farms
                    WHERE farmer_id = :farmer_id
                    ORDER BY farm_id
                """),
                {"farmer_id": current_user.get("farmer_id")}
            )

        farms = [dict(row._mapping) for row in result]

    return farms


@app.post("/crops")
def create_crop(
    crop: Crop,
    current_user: dict = Depends(get_current_user)
):
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT farmer_id
                FROM farms
                WHERE farm_id = :farm_id
            """),
            {"farm_id": crop.farm_id}
        )
        farm_owner = result.fetchone()

    if farm_owner is None:
        raise HTTPException(
            status_code=404,
            detail="Farm not found"
        )

    if (
        current_user.get("role") != "admin"
        and current_user.get("farmer_id") != farm_owner.farmer_id
    ):
        raise HTTPException(
            status_code=403,
            detail="You can only add crops to your own farms"
        )

    with engine.begin() as connection:
        result = connection.execute(
            text("""
                INSERT INTO crops
                (
                    farm_id, crop_name, variety, season,
                    sowing_date, harvest_date, previous_yield, yield_unit
                )
                VALUES
                (
                    :farm_id, :crop_name, :variety, :season,
                    :sowing_date, :harvest_date, :previous_yield, :yield_unit
                )
                RETURNING crop_id
            """),
            crop.model_dump()
        )
        crop_id = result.scalar_one()

    return {
        "message": "Crop created successfully",
        "crop_id": crop_id,
        "crop": crop
    }


@app.get("/crops")
def get_crops(
    current_user: dict = Depends(get_current_user)
):
    with engine.connect() as connection:
        if current_user.get("role") == "admin":
            result = connection.execute(
                text("SELECT * FROM crops ORDER BY crop_id")
            )
        else:
            result = connection.execute(
                text("""
                    SELECT c.*
                    FROM crops c
                    JOIN farms f ON c.farm_id = f.farm_id
                    WHERE f.farmer_id = :farmer_id
                    ORDER BY c.crop_id
                """),
                {"farmer_id": current_user.get("farmer_id")}
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

@app.post("/documents")
async def upload_document(
    farmer_id: int = Form(...),
    document_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "admin":
        if current_user["farmer_id"] != farmer_id:
            raise HTTPException(
                status_code=403,
                detail="You can upload documents only for your own profile"
            )

    original_name = Path(file.filename or "document").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, JPG, JPEG, and PNG files are allowed"
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 5 MB"
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty"
        )

    saved_name = f"{uuid4().hex}{extension}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        saved_path.write_bytes(contents)

        with engine.begin() as connection:
            result = connection.execute(
                text("""
                    INSERT INTO farmer_documents
                    (farmer_id, document_type, document_name, document_path)
                    VALUES
                    (:farmer_id, :document_type, :document_name, :document_path)
                    RETURNING document_id, farmer_id, document_type,
                              document_name, document_path, uploaded_at, verified
                """),
                {
                    "farmer_id": farmer_id,
                    "document_type": document_type,
                    "document_name": original_name,
                    "document_path": str(Path("uploads") / saved_name)
                }
            )
            document = result.fetchone()

        return {
            "message": "Document uploaded successfully",
            "document": dict(document._mapping)
        }

    except Exception:
        saved_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()


@app.patch("/documents/{document_id}/verify")
def verify_document(
    document_id: int,
    current_user: dict = Depends(require_admin)
):
    with engine.begin() as connection:
        result = connection.execute(
            text("""
                UPDATE farmer_documents
                SET verified = TRUE
                WHERE document_id = :document_id
                RETURNING *
            """),
            {"document_id": document_id}
        )

        document = result.fetchone()

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return {
        "message": "Document verified successfully",
        "document": dict(document._mapping)
    }

@app.get("/farmers/{farmer_id}/documents")
def get_farmer_documents(
    farmer_id: int,
    current_user: dict = Depends(get_current_user)
):
    if (current_user.get("role") != "admin" and current_user.get("farmer_id") != farmer_id):
        raise HTTPException(
            status_code=403,
            detail="You can only access your own documents"
        )

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM farmer_documents
                WHERE farmer_id = :farmer_id
                ORDER BY uploaded_at DESC
            """),
            {"farmer_id": farmer_id}
        )

        documents = [dict(row._mapping) for row in result]

    return documents

@app.get("/farmers/{farmer_id}/documents/verified")
def get_verified_documents(
    farmer_id: int,
    current_user: dict = Depends(get_current_user)
):
    if (current_user.get("role") != "admin" and current_user.get("farmer_id") != farmer_id):
        raise HTTPException(
            status_code=403,
            detail="You can only access your own documents"
        )

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT *
                FROM farmer_documents
                WHERE farmer_id = :farmer_id
                AND verified = TRUE
            """),
            {"farmer_id": farmer_id}
        )

        documents = [dict(row._mapping) for row in result]

    return {
        "farmer_id": farmer_id,
        "verified_documents": documents
    }

@app.post("/predict-crops")
def predict_crop_recommendation(data: CropPredictionInput, current_user: dict = Depends(get_current_user)):

    farmer_data = data.model_dump()

    recommendations = predict_crops(farmer_data)

    return {
        "recommendations": recommendations
    }

@app.post("/recommend-schemes")
def recommend_government_schemes(data: SchemeRecommendationInput, current_user: dict = Depends(get_current_user)):

    farmer_data = data.model_dump()

    recommendations = recommend_all_schemes(farmer_data)

    return {
        "recommendations": recommendations
    }


@app.get("/farmers/{farmer_id}/recommend-schemes")
def recommend_schemes_for_farmer(
    farmer_id: int,
    current_user: dict = Depends(get_current_user)
):
    if (current_user.get("role") != "admin" and current_user.get("farmer_id") != farmer_id):
        raise HTTPException(
            status_code=403,
            detail="You can only access your own scheme recommendations"
        )

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

    # Check for verified land ownership document
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM farmer_documents
                WHERE farmer_id = :farmer_id
                AND document_type = 'Land Ownership'
                AND verified = TRUE
            """),
            {"farmer_id": farmer_id}
        )

        verified_land_document = result.scalar() > 0

    # Land ownership is verified only if both
    # the profile and the verified document confirm it.
    farmer_data["owns_cultivable_land"] = (
        farmer_data.get("owns_cultivable_land", False)
        and verified_land_document
    )

    recommendations = recommend_all_schemes(farmer_data)

    return {
        "farmer_id": farmer_id,
        "recommendations": recommendations
    }
