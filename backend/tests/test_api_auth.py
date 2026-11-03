"""Integration tests for auth endpoints and protected route guard."""


def test_register_and_login(client):
    r = client.post("/api/v1/auth/register", json={
        "email": "user@test.com", "password": "testtest", "full_name": "Test User",
    })
    assert r.status_code == 201
    assert r.json()["email"] == "user@test.com"

    r = client.post("/api/v1/auth/login", json={
        "email": "user@test.com", "password": "testtest",
    })
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_duplicate_email_rejected(client):
    payload = {"email": "dup@test.com", "password": "testtest", "full_name": "A"}
    client.post("/api/v1/auth/register", json=payload)
    r2 = client.post("/api/v1/auth/register", json=payload)
    assert r2.status_code == 400


def test_wrong_password_rejected(client):
    client.post("/api/v1/auth/register", json={"email": "wp@test.com", "password": "testtest"})
    r = client.post("/api/v1/auth/login", json={"email": "wp@test.com", "password": "wrong"})
    assert r.status_code == 401


def test_protected_route_requires_auth(client):
    r = client.get("/api/v1/documents")
    assert r.status_code == 401


def test_me_returns_user(authed_client):
    r = authed_client.get("/api/v1/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == "tester@example.com"


def test_token_refresh(client):
    client.post("/api/v1/auth/register", json={"email": "rf@test.com", "password": "testtest"})
    login = client.post("/api/v1/auth/login", json={"email": "rf@test.com", "password": "testtest"})
    refresh_token = login.json()["refresh_token"]
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_dashboard_requires_auth(client):
    r = client.get("/api/v1/stats/dashboard")
    assert r.status_code == 401


def test_dashboard_accessible_when_authed(authed_client):
    r = authed_client.get("/api/v1/stats/dashboard")
    assert r.status_code == 200
    data = r.json()
    assert "total_documents" in data
    assert "total_tokens" in data


def test_health_public(client):
    r = client.get("/api/v1/stats/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_document_requires_auth(client):
    import io
    r = client.post("/api/v1/documents", files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")})
    assert r.status_code == 401
