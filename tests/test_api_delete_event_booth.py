"""Regression and unit tests for issue #786:
- delete_event_booth route shadowing and signature bug
- _require_access and validate_production_secrets token enforcement
"""

import pytest
from fastapi.testclient import TestClient

from fastapi_app import app
from portal.auth import create_user_token
from portal.config import settings
from portal.database import create_booth, create_event, create_room, get_session
from portal.models import DBBooth

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    import anyio

    from portal.database import configure, dispose, init_db

    configure("sqlite+aiosqlite://")
    anyio.run(init_db)
    yield
    anyio.run(dispose)


async def _seed_test_event_and_booths():
    async with get_session() as session:
        ev = await create_event(session, slug="event-786", display_name="Event 786")
        r_int = await create_room(session, event_id=ev.id, display_name="Room Int", eventyay_room_id="ey-int-1")
        r_str = await create_room(
            session, event_id=ev.id, display_name="Room Str", eventyay_room_id="eventyay-room-abc"
        )

        # Booth for int room
        b1 = DBBooth(event_id=ev.id, room_id=r_int.id, language_code="en", language_name="English")
        # Booth for eventyay string room
        b2 = DBBooth(event_id=ev.id, room_id=r_str.id, language_code="de", language_name="German")
        session.add_all([b1, b2])
        await session.commit()
        return ev.id, r_int.id, r_str.eventyay_room_id


def test_delete_booth_by_language_int_routing(monkeypatch):
    """Integer room_id routes to delete_booth_by_language and deletes booth."""
    import anyio

    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")
    ev_id, r_int_id, _ = anyio.run(_seed_test_event_and_booths)

    token = create_user_token(user_id=1, email="admin@test.com", is_admin=True)
    res = client.delete(
        f"/api/events/event-786/rooms/{r_int_id}/booths/en",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 204


def test_delete_event_booth_string_routing_and_deletion(monkeypatch):
    """String eventyay_room_id routes to delete_event_booth without 422 or 500 error."""
    import anyio

    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")
    ev_id, _, ey_room_id = anyio.run(_seed_test_event_and_booths)

    # Valid Bearer user token
    token = create_user_token(user_id=1, email="admin@test.com", is_admin=True)
    res = client.delete(
        f"/api/events/event-786/rooms/{ey_room_id}/booths/de",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 204


def test_delete_event_booth_with_query_token(monkeypatch):
    """delete_event_booth succeeds with query token."""
    import anyio

    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")
    ev_id, _, ey_room_id = anyio.run(_seed_test_event_and_booths)

    res = client.delete(f"/api/events/event-786/rooms/{ey_room_id}/booths/de?token=secret-test-token")
    assert res.status_code == 204


def test_delete_event_booth_unauthorized_when_token_configured(monkeypatch):
    """delete_event_booth rejects invalid credentials with 403."""
    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")

    res = client.delete(
        "/api/events/event-786/rooms/any-room/booths/de",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert res.status_code == 403


def test_require_access_rejected_in_production_when_token_unset(monkeypatch):
    """In production (debug=False), unconfigured booth_access_token rejects with 403."""
    from fastapi import Request

    from portal.utils import _require_access

    monkeypatch.setattr(settings, "booth_access_token", "")
    monkeypatch.setattr(settings, "debug", False)

    req = Request(scope={"type": "http", "method": "DELETE", "path": "/"})
    with pytest.raises(Exception) as exc_info:
        _require_access(req, credentials=None, token_query=None)
    assert "BOOTH_ACCESS_TOKEN is not configured" in str(exc_info.value)


def test_validate_production_secrets_checks_booth_access_token(monkeypatch):
    """validate_production_secrets raises RuntimeError if booth_access_token is empty in production."""
    monkeypatch.setattr(settings, "debug", False)
    monkeypatch.setattr(settings, "secret_key", "valid-strong-secret-key-32-chars-long!")
    monkeypatch.setattr(settings, "mediamtx_auth_hook_secret", "valid-strong-hook-secret-32-chars!")
    monkeypatch.setattr(settings, "booth_access_token", "")

    with pytest.raises(RuntimeError) as exc_info:
        settings.validate_production_secrets()
    assert "BOOTH_ACCESS_TOKEN is not set" in str(exc_info.value)
