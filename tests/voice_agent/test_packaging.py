from fastapi.testclient import TestClient

from voice_agent.server import create_app


def test_healthz_reports_service_status():
    client = TestClient(create_app())
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dockerfile_does_not_embed_secrets_or_model_weights():
    content = open("Dockerfile", encoding="utf-8").read()

    assert "COPY voice_agent" in content
    assert "ENV OPENAI_COMPATIBLE_API_KEY" not in content
    assert ".cache" not in content
