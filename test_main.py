from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app instance

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Lease End DSPy API is running."}

def test_generate_reply():
    payload = {
        "comment": "I love my leased car but the buyout seems too expensive.",
        "postId": "12345"
    }
    response = client.post("/generate-reply", json=payload)
    assert response.status_code == 200
    response_data = response.json()
    assert "reply" in response_data
    assert "postId" in response_data
    assert response_data["postId"] == "12345"
    # Check that reply is not empty and not an error message
    assert len(response_data["reply"]) > 0
    assert "error" not in response_data or response_data.get("error") is None