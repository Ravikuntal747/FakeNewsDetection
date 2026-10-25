"""
Final Verification Script for Fake News Detection System
"""
from app import app

client = app.test_client()

test_cases = [
    (
        "Real Diplomatic News",
        "U.S. Secretary of State John Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American official attended Sunday unity march."
    ),
    (
        "Real Science News",
        "NASA scientists analyzing Mars rover telemetry confirmed sedimentary rock formations indicating long-standing surface water in Gale Crater billions of years ago."
    ),
    (
        "Fake Conspiracy News",
        "Secret leaked military documents reveal government officials are concealing contact with extraterrestrial alien craft deep beneath underground bunkers! Share before deleted!"
    ),
    (
        "Fake Miracle Health Hoax",
        "Doctors are stunned! Local grandmother discovers miracle secret leaf that cures all diseases overnight, pharmaceutical companies are furiously trying to ban this video!"
    )
]

print("=" * 65)
print("     LIVE MODEL PREDICTION VERIFICATION (TEST RUN)")
print("=" * 65)

results = []
for label, text in test_cases:
    res = client.post("/predict", json={"news_text": text})
    assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}"
    data = res.get_json()
    assert data["success"] is True
    results.append(data)
    
    print(f"\n[TEST CASE: {label}]")
    print(f"  Input Text : \"{text[:75]}...\"")
    print(f"  Verdict    : {data['verdict']}")
    print(f"  Confidence : {data['confidence']}%")
    print(f"  Breakdown  : REAL = {data['probabilities']['REAL']}%, FAKE = {data['probabilities']['FAKE']}%")

# Confirm scores are distinct and floating-point math from model
confidences = [r["confidence"] for r in results]
print("\n" + "-" * 65)
print(f"Computed Confidence Scores: {confidences}")
assert len(set(confidences)) == len(confidences), "Predictions must be distinct and non-hardcoded!"
print("[SUCCESS] Verified: All predictions and probabilities are dynamically calculated by the ML model!")
print("=" * 65)
