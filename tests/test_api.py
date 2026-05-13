"""
tests/test_api.py — Comprehensive API tests for the Medical Healthcare Chatbot.

28 tests covering:
  Health (2) · Analytics/statistics (6) · Patient list filters (8)
  Patient detail (3) · Chat RAG (4) · LoS prediction (4)
  Medication search (3) · CORS (1)

Run:
  pytest tests/test_api.py -v -p no:cacheprovider
"""

import sys
import types
from pathlib import Path
import re

# ── Fake RAG pipeline (bypass heavy ML imports in CI) ────────────────────────
class _FakePipeline:
    def chat(self, query):
        intent = "medication_query" if "medication" in query.lower() else "analytics_query"
        return {
            "answer": f"Processed: {query[:40]}",
            "intent": intent,
            "sources": [
                {"patient_id": "1", "doc_type": "patient_record", "snippet": "Patient 1 data..."},
                {"patient_id": "2", "doc_type": "patient_record", "snippet": "Patient 2 data..."},
            ],
            "source_count": 2,
        }

    def predict_los(self, features):
        days = 25.0 if features.get("icu_admission", 0) == 1 else 10.5
        if features.get("intubated", 0) == 1:
            days = days + 8.0
        return {
            "predicted_los_days": round(days, 1),
            "confidence": "High (R2=0.9915)",
            "risk_level": "High" if days > 20 else "Moderate",
        }

    def search_medications(self, query, k=5):
        meds = [
            {"name": "rosuvastatin",  "content": "Medication: rosuvastatin"},
            {"name": "metformin",     "content": "Medication: metformin"},
            {"name": "lisinopril",    "content": "Medication: lisinopril"},
            {"name": "dexamethasone", "content": "Medication: dexamethasone"},
            {"name": "remdesivir",    "content": "Medication: remdesivir"},
            {"name": "heparin",       "content": "Medication: heparin"},
            {"name": "azithromycin",  "content": "Medication: azithromycin"},
        ]
        return meds[:k]


# ── Inject fakes BEFORE any real imports touch rag ────────────────────────────
_fake_rag_pipeline_mod = types.ModuleType("rag.pipeline")
_fake_rag_pipeline_mod.MedicalRAGPipeline = _FakePipeline

def _get_pipeline_fn():
    return _FakePipeline()

_fake_rag_pipeline_mod.get_pipeline = _get_pipeline_fn
_fake_rag_mod = types.ModuleType("rag")
_fake_rag_mod.pipeline = _fake_rag_pipeline_mod
sys.modules["rag"] = _fake_rag_mod
sys.modules["rag.pipeline"] = _fake_rag_pipeline_mod
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from deployment.api import app

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Health endpoint (2 tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_health_check():
    """API returns status=ok with correct version."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "2.0.0"


def test_health_has_utc_timestamp():
    """Health response must include an ISO-8601 UTC timestamp."""
    r = client.get("/health")
    ts = r.json().get("timestamp", "")
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts), f"Bad timestamp: {ts}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Dataset statistics / analytics (6 tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_analytics_total_patients():
    """Dataset must contain exactly 508 patients."""
    r = client.get("/api/analytics/summary")
    assert r.status_code == 200
    assert r.json()["total_patients"] == 508


def test_analytics_icu_ward_counts():
    """ICU + Ward admissions must sum to total patients."""
    r = client.get("/api/analytics/summary")
    d = r.json()
    assert d["icu_admissions"] + d["ward_admissions"] == d["total_patients"]


def test_analytics_icu_percentage():
    """ICU percentage should equal icu_admissions / total * 100 (+-0.2)."""
    r = client.get("/api/analytics/summary")
    d = r.json()
    expected = round(d["icu_admissions"] / d["total_patients"] * 100, 1)
    assert abs(d["icu_pct"] - expected) <= 0.2


def test_analytics_avg_hospital_los_is_positive():
    """Average hospital length-of-stay must be a positive number."""
    r = client.get("/api/analytics/summary")
    avg = r.json().get("avg_hospital_los_days")
    assert avg is not None and avg > 0, f"Expected positive avg LoS, got {avg}"


def test_analytics_avg_hospital_los_in_realistic_range():
    """Average hospital LoS should be between 1 and 60 days."""
    r = client.get("/api/analytics/summary")
    avg = r.json()["avg_hospital_los_days"]
    assert 1.0 <= avg <= 60.0, f"Avg LoS {avg} outside expected range"


def test_analytics_response_has_all_keys():
    """Analytics response must include all 7 required keys."""
    required = {
        "total_patients", "icu_admissions", "ward_admissions",
        "icu_pct", "avg_hospital_los_days", "avg_icu_los_days", "mortality_pct",
    }
    r = client.get("/api/analytics/summary")
    missing = required - set(r.json().keys())
    assert not missing, f"Missing keys: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Patient list: pagination and filters (8 tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_patient_list_default_returns_20():
    """Default page returns 20 patients from 508 total."""
    r = client.get("/api/patients")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 508
    assert len(d["patients"]) == 20
    assert d["page"] == 1
    assert d["page_size"] == 20


def test_patient_list_custom_page_size():
    """Page size param correctly limits results."""
    r = client.get("/api/patients?page=1&page_size=5")
    assert r.status_code == 200
    d = r.json()
    assert len(d["patients"]) == 5
    assert d["page_size"] == 5


def test_patient_list_pagination_page2():
    """Page 2 returns different patients than page 1."""
    r1 = client.get("/api/patients?page=1&page_size=5")
    r2 = client.get("/api/patients?page=2&page_size=5")
    ids1 = [p.get("id") for p in r1.json()["patients"]]
    ids2 = [p.get("id") for p in r2.json()["patients"]]
    assert ids1 != ids2, "Pages 1 and 2 should not return the same patients"


def test_patient_list_filter_icu():
    """Filter ICU: all returned patients must have ICU disposition."""
    r = client.get("/api/patients?disposition=ICU&page_size=100")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 43, f"Expected 43 ICU patients, got {d['total']}"
    for p in d["patients"]:
        assert p["admission_disposition"].upper() == "ICU"


def test_patient_list_filter_ward():
    """Filter WARD: all returned patients must have WARD disposition."""
    r = client.get("/api/patients?disposition=WARD&page_size=100")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 465, f"Expected 465 WARD patients, got {d['total']}"
    for p in d["patients"]:
        assert p["admission_disposition"].upper() == "WARD"


def test_patient_list_filter_sex_male():
    """Filter sex=Male should return only male patients."""
    r = client.get("/api/patients?sex=Male&page_size=50")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] > 0
    for p in d["patients"]:
        assert p["sex"].lower() in ("male", "m"), f"Unexpected sex: {p['sex']}"


def test_patient_list_filter_sex_female():
    """Filter sex=Female should return only female patients."""
    r = client.get("/api/patients?sex=Female&page_size=50")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] > 0
    for p in d["patients"]:
        assert p["sex"].lower() in ("female", "f"), f"Unexpected sex: {p['sex']}"


def test_patient_list_male_plus_female_equals_total():
    """Male + female counts must sum to total patients."""
    male   = client.get("/api/patients?sex=Male").json()["total"]
    female = client.get("/api/patients?sex=Female").json()["total"]
    total  = client.get("/api/patients").json()["total"]
    assert male + female == total, f"Male({male}) + Female({female}) != Total({total})"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Patient detail (3 tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_patient_detail_found():
    """Fetching patient 1 returns admission + outcome profile."""
    r = client.get("/api/patients/1")
    assert r.status_code == 200
    d = r.json()
    assert d["patient_id"] == 1
    assert "admission" in d and "outcome" in d
    assert isinstance(d["admission"], dict)


def test_patient_detail_has_expected_admission_keys():
    """Patient admission dict must include id, age, sex fields."""
    r = client.get("/api/patients/1")
    adm = r.json()["admission"]
    for key in ("id", "age", "sex"):
        assert key in adm, f"Expected key '{key}' in admission dict"


def test_patient_detail_not_found():
    """Non-existent patient ID must return 404."""
    assert client.get("/api/patients/999999").status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Chat RAG endpoint (4 tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_returns_required_keys():
    """Chat response must include all 5 required fields."""
    r = client.post("/api/chat", json={"message": "What is the average hospital length of stay?"})
    assert r.status_code == 200
    d = r.json()
    for key in ("answer", "intent", "sources", "source_count", "timestamp"):
        assert key in d, f"Missing key: {key}"


def test_chat_timestamp_is_iso8601():
    """Chat timestamp must be a valid ISO-8601 string."""
    r = client.post("/api/chat", json={"message": "How many ICU patients are there?"})
    ts = r.json().get("timestamp", "")
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts), f"Invalid timestamp: {ts}"


def test_chat_medication_intent():
    """Messages about medications trigger medication_query intent."""
    r = client.post("/api/chat", json={"message": "What medications are commonly prescribed?"})
    assert r.status_code == 200
    assert r.json()["intent"] == "medication_query"


def test_chat_with_session_id():
    """Session ID field is accepted without errors."""
    r = client.post("/api/chat", json={
        "message": "Tell me about patient demographics.",
        "session_id": "test-session-abc",
    })
    assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — LoS Prediction (4 tests)
# ─────────────────────────────────────────────────────────────────────────────

_WARD_PAYLOAD = {
    "age": 65.0, "sex": 0,
    "systolic_blood_pressure": 130.0, "diastolic_blood_pressure": 80.0,
    "heart_rate": 90.0, "respiratory_rate": 18.0,
    "oxygen_saturation": 95.0, "temperature": 37.5,
    "icu_admission": 0, "intubated": 0,
}

def test_los_prediction_ward_patient():
    """Valid WARD patient returns predicted days and confidence."""
    r = client.post("/api/predict/los", json=_WARD_PAYLOAD)
    assert r.status_code == 200
    d = r.json()
    assert "predicted_los_days" in d
    assert "confidence" in d
    assert d["predicted_los_days"] > 0


def test_los_prediction_icu_longer_than_ward():
    """ICU patients should have a higher predicted LoS than WARD patients."""
    ward_days = client.post("/api/predict/los", json=_WARD_PAYLOAD).json()["predicted_los_days"]
    icu_payload = dict(_WARD_PAYLOAD)
    icu_payload["icu_admission"] = 1
    icu_days = client.post("/api/predict/los", json=icu_payload).json()["predicted_los_days"]
    assert icu_days > ward_days, f"ICU({icu_days}) should exceed WARD({ward_days})"


def test_los_prediction_intubated_increases_stay():
    """Intubated patients should have a higher predicted LoS."""
    base = client.post("/api/predict/los", json=_WARD_PAYLOAD).json()["predicted_los_days"]
    intubated_payload = dict(_WARD_PAYLOAD)
    intubated_payload["intubated"] = 1
    intubated = client.post("/api/predict/los", json=intubated_payload).json()["predicted_los_days"]
    assert intubated > base, f"Intubated({intubated}) should exceed baseline({base})"


def test_los_prediction_missing_field_returns_422():
    """Incomplete payload must return HTTP 422."""
    r = client.post("/api/predict/los", json={"age": 65.0})
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Medication semantic search (3 tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_medication_search_returns_results():
    """Medication search returns a list with name field."""
    r = client.get("/api/medications/search?q=statin&k=3")
    assert r.status_code == 200
    d = r.json()
    assert "results" in d
    assert len(d["results"]) <= 3
    for item in d["results"]:
        assert "name" in item


def test_medication_search_respects_k_param():
    """k parameter caps the number of returned medications."""
    for k in (1, 3, 5):
        r = client.get(f"/api/medications/search?q=antibiotic&k={k}")
        assert r.status_code == 200
        assert len(r.json()["results"]) <= k, f"k={k} exceeded"


def test_medication_search_short_query_rejected():
    """Query shorter than 2 characters must be rejected (HTTP 422)."""
    r = client.get("/api/medications/search?q=a")
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — HTTP headers / CORS (1 test)
# ─────────────────────────────────────────────────────────────────────────────

def test_cors_header_present():
    """CORS allow-origin header must be present on API responses."""
    r = client.get("/health", headers={"Origin": "http://localhost:8501"})
    header_keys = {h.lower() for h in r.headers.keys()}
    assert "access-control-allow-origin" in header_keys, "CORS header missing"
