import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCHEMES_FILE = PROJECT_ROOT / "data" / "government_schemes.json"
FARMER_FILE = PROJECT_ROOT / "data" / "farmer_profile.json"

with open(SCHEMES_FILE, "r") as file:
    schemes = json.load(file)

with open(FARMER_FILE, "r") as file:
    farmer = json.load(file)

print("Schemes loaded:", len(schemes))
print("Farmer profile loaded:", farmer)


def recommend_scheme(scheme, farmer):
    rules = scheme["eligibility_rules"]

    failed_conditions = []
    missing_requirements = []

    # PM-KISAN
    if scheme["short_name"] == "PM-KISAN":

        if farmer["owns_cultivable_land"] != rules["owns_cultivable_land"]:
            failed_conditions.append("Land ownership condition")
            missing_requirements.append("Proof of cultivable land ownership")

        if farmer["income_tax_payer"] != rules["income_tax_payer"]:
            failed_conditions.append("Income tax condition")
            missing_requirements.append("Income tax eligibility verification")

        if farmer["government_employee"]:
            failed_conditions.append("Government employment condition")
            missing_requirements.append("Government employment eligibility verification")

        if farmer["monthly_pension"] >= 10000:
            failed_conditions.append("Pension condition")
            missing_requirements.append("Pension eligibility verification")

        if farmer["practicing_professional"] != rules["practicing_professional"]:
            failed_conditions.append("Professional status condition")
            missing_requirements.append("Professional status verification")

        if farmer["nri_farmer"] != rules["nri_farmer"]:
            failed_conditions.append("NRI status condition")
            missing_requirements.append("Residency eligibility verification")

    # PMFBY
    elif scheme["short_name"] == "PMFBY":

        if farmer["grows_notified_crop"] != rules["grows_notified_crop"]:
            failed_conditions.append("Notified crop condition")
            missing_requirements.append("Proof of notified crop")

        if farmer["in_notified_area"] != rules["in_notified_area"]:
            failed_conditions.append("Notified area condition")
            missing_requirements.append("Proof that the farm is in a notified area")

        if farmer["has_insurable_interest"] != rules["has_insurable_interest"]:
            failed_conditions.append("Insurable interest condition")
            missing_requirements.append("Proof of insurable interest")

        if farmer["has_required_documents"] != rules["has_required_documents"]:
            failed_conditions.append("Required documents condition")
            missing_requirements.append("Required land/tenancy and crop documents")

    # KCC
    elif scheme["short_name"] == "KCC":

        if farmer["is_farmer"] != rules["is_farmer"]:
            failed_conditions.append("Farmer status condition")
            missing_requirements.append("Proof of farmer status")

        if farmer["has_cultivation_activity"] != rules["has_cultivation_activity"]:
            failed_conditions.append("Cultivation activity condition")
            missing_requirements.append("Proof of cultivation activity")

    # Soil Health Card
    elif scheme["short_name"] == "SHC":

        if farmer["is_farmer"] != rules["is_farmer"]:
            failed_conditions.append("Farmer status condition")
            missing_requirements.append("Proof of farmer status")

        if farmer["requires_soil_testing"] != rules["requires_soil_testing"]:
            failed_conditions.append("Soil testing condition")
            missing_requirements.append("Soil sample for testing")

    # Krishi Bhagya
    elif scheme["short_name"] == "KRISHI_BHAGYA":

        if farmer["is_farmer"] != rules["is_farmer"]:
            failed_conditions.append("Farmer status condition")
            missing_requirements.append("Proof of farmer status")

        if farmer["practices_rainfed_agriculture"] != rules["practices_rainfed_agriculture"]:
            failed_conditions.append("Rainfed agriculture condition")
            missing_requirements.append("Proof of rainfed agriculture")

    # Final recommendation
    if len(failed_conditions) == 0:
        return {
            "scheme_name": scheme["scheme_name"],
            "status": "ELIGIBLE",
            "reason": scheme["recommendation_reason"],
            "next_action": scheme["next_action"],
            "failed_conditions": [],
            "missing_requirements": []
        }

    return {
        "scheme_name": scheme["scheme_name"],
        "status": "NOT ELIGIBLE",
        "reason": "One or more eligibility conditions are not satisfied.",
        "next_action": "Review the failed eligibility conditions before applying.",
        "failed_conditions": failed_conditions,
        "missing_requirements": missing_requirements
    }


# Check every scheme
for scheme in schemes:
    result = recommend_scheme(scheme, farmer)

    print("\nScheme:", result["scheme_name"])
    print("Recommendation:", result["status"])
    print("Reason:", result["reason"])
    print("Next action:", result["next_action"])

    if result["failed_conditions"]:
        print("Failed conditions:", result["failed_conditions"])

    if result["missing_requirements"]:
        print("Missing requirements:", result["missing_requirements"])

# Prioritize eligible schemes

priority_scores = {
    "Pradhan Mantri Kisan Samman Nidhi": 5,
    "Pradhan Mantri Fasal Bima Yojana": 4,
    "Kisan Credit Card": 3,
    "Soil Health Card Scheme": 2,
    "Krishi Bhagya": 1
}

eligible_schemes = []

for scheme in schemes:
    result = recommend_scheme(scheme, farmer)

    if result["status"] == "ELIGIBLE":
        eligible_schemes.append({
            "scheme_name": result["scheme_name"],
            "priority": priority_scores[result["scheme_name"]],
            "next_action": result["next_action"]
        })

eligible_schemes.sort(
    key=lambda x: x["priority"],
    reverse=True
)

print("\n===== PRIORITIZED SCHEMES =====")

for scheme in eligible_schemes:
    print(
        scheme["priority"],
        "-",
        scheme["scheme_name"]
    )
    print("  Next action:", scheme["next_action"])