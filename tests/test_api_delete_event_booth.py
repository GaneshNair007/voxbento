from __future__ import annotations

# Regression and unit tests for issue #786:
# - delete_event_booth route shadowing and signature bug
# - _require_access and validate_production_secrets token enforcement
# - Event-scoped authorization and BOLA protection
# - Numeric external room ID disambiguation and collision avoidance
# - Idempotent concurrent DELETE handling
# - Token redaction in access logs for API paths
import logging

import anyio
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import select

from fastapi_app import _UvicornTokenRedactor, app
from portal.auth import create_user_token
from portal.config import settings
from portal.database import (
    create_event,
    create_room,
    create_user,
    get_session,
    set_event_membership,
)
from portal.models import DBBooth
from portal.utils import _require_access

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
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

        admin = await create_user(
            session, email="admin@test.com", display_name="Admin", password_hash="hash", is_admin=True
        )
        await session.commit()
        return ev.id, r_int.id, r_str.eventyay_room_id, admin.id


def test_delete_booth_by_language_int_routing(monkeypatch):
    """Integer room_id routes to delete_booth_by_language and confirms database deletion."""
    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")
    ev_id, r_int_id, _, admin_id = anyio.run(_seed_test_event_and_booths)

    token = create_user_token(user_id=admin_id, email="admin@test.com", is_admin=True)
    res = client.delete(
        f"/api/events/event-786/rooms/{r_int_id}/booths/en",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 204

    # Assert row is deleted from database
    async def _check_db():
        async with get_session() as session:
            booth = await session.scalar(
                select(DBBooth).where(
                    DBBooth.event_id == ev_id,
                    DBBooth.room_id == r_int_id,
                    DBBooth.language_code == "en",
                )
            )
            assert booth is None

    anyio.run(_check_db)


def test_delete_event_booth_string_routing_and_deletion(monkeypatch):
    """String eventyay_room_id routes to delete_event_booth and confirms database deletion."""
    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")
    ev_id, _, ey_room_id, admin_id = anyio.run(_seed_test_event_and_booths)

    token = create_user_token(user_id=admin_id, email="admin@test.com", is_admin=True)
    res = client.delete(
        f"/api/events/event-786/rooms/{ey_room_id}/booths/de",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 204

    # Assert row is deleted from database
    async def _check_db():
        async with get_session() as session:
            booth = await session.scalar(
                select(DBBooth).where(
                    DBBooth.event_id == ev_id,
                    DBBooth.language_code == "de",
                )
            )
            assert booth is None

    anyio.run(_check_db)


def test_delete_event_booth_with_query_token(monkeypatch):
    """delete_event_booth succeeds with query token and deletes the row."""
    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")
    ev_id, _, ey_room_id, _ = anyio.run(_seed_test_event_and_booths)

    res = client.delete(f"/api/events/event-786/rooms/{ey_room_id}/booths/de?token=secret-test-token")
    assert res.status_code == 204

    async def _check_db():
        async with get_session() as session:
            booth = await session.scalar(
                select(DBBooth).where(DBBooth.event_id == ev_id, DBBooth.language_code == "de")
            )
            assert booth is None

    anyio.run(_check_db)


def test_delete_event_booth_with_bearer_master_token(monkeypatch):
    """delete_event_booth succeeds with BOOTH_ACCESS_TOKEN in Bearer header without query param."""
    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")
    ev_id, _, ey_room_id, _ = anyio.run(_seed_test_event_and_booths)

    res = client.delete(
        f"/api/events/event-786/rooms/{ey_room_id}/booths/de",
        headers={"Authorization": "Bearer secret-test-token"},
    )
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
    monkeypatch.setattr(settings, "booth_access_token", "")
    monkeypatch.setattr(settings, "debug", False)

    req = Request(scope={"type": "http", "method": "DELETE", "path": "/"})
    with pytest.raises(Exception) as exc_info:
        _require_access(req, credentials=None, token_query=None)
    assert "BOOTH_ACCESS_TOKEN is not configured" in str(exc_info.value)


def test_validate_production_secrets_rejects_weak_booth_access_token(monkeypatch):
    """validate_production_secrets raises RuntimeError if booth_access_token is a weak default."""
    monkeypatch.setattr(settings, "debug", False)
    monkeypatch.setattr(settings, "secret_key", "valid-strong-secret-key-32-chars-long!")
    monkeypatch.setattr(settings, "mediamtx_auth_hook_secret", "valid-strong-hook-secret-32-chars!")
    monkeypatch.setattr(settings, "booth_access_token", "change-me")

    with pytest.raises(RuntimeError) as exc_info:
        settings.validate_production_secrets()
    assert "BOOTH_ACCESS_TOKEN is set to a known-weak default value" in str(exc_info.value)


def test_cross_event_unauthorized_deletion_blocked(monkeypatch):
    """Non-admin user who owns Event B cannot delete booths in Event A (CWE-863 BOLA)."""
    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")

    async def _setup_two_events():
        async with get_session() as session:
            ev_a = await create_event(session, slug="event-a", display_name="Event A")
            ev_b = await create_event(session, slug="event-b", display_name="Event B")

            r_a = await create_room(session, event_id=ev_a.id, display_name="Room A", eventyay_room_id="ey-a")
            r_b = await create_room(session, event_id=ev_b.id, display_name="Room B", eventyay_room_id="ey-b")

            b_a = DBBooth(event_id=ev_a.id, room_id=r_a.id, language_code="fr", language_name="French")
            b_b = DBBooth(event_id=ev_b.id, room_id=r_b.id, language_code="es", language_name="Spanish")
            session.add_all([b_a, b_b])

            user_b = await create_user(
                session, email="owner_b@test.com", display_name="User B", password_hash="hash", is_admin=False
            )
            await set_event_membership(session, user_id=user_b.id, event_id=ev_b.id, role="event_owner")
            await session.commit()
            return ev_a.id, r_a.id, user_b.id

    ev_a_id, r_a_id, user_b_id = anyio.run(_setup_two_events)

    user_b_token = create_user_token(user_id=user_b_id, email="owner_b@test.com", is_admin=False)

    res = client.delete(
        f"/api/events/event-a/rooms/{r_a_id}/booths/fr",
        headers={"Authorization": f"Bearer {user_b_token}"},
    )
    assert res.status_code == 403
    assert "Forbidden" in res.json()["detail"]

    async def _verify_booth_retained():
        async with get_session() as session:
            booth = await session.scalar(
                select(DBBooth).where(
                    DBBooth.event_id == ev_a_id,
                    DBBooth.room_id == r_a_id,
                    DBBooth.language_code == "fr",
                )
            )
            assert booth is not None

    anyio.run(_verify_booth_retained)


def test_numeric_external_room_id_collision_handled(monkeypatch):
    """Numeric external eventyay_room_id resolves external room without deleting another room's booth."""
    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")

    async def _setup_colliding_rooms():
        async with get_session() as session:
            ev = await create_event(session, slug="collision-event", display_name="Collision Event")
            r1 = await create_room(session, event_id=ev.id, display_name="Internal Room 1", eventyay_room_id="ey-alpha")
            r2 = await create_room(session, event_id=ev.id, display_name="Target Room 2", eventyay_room_id=str(r1.id))

            b1 = DBBooth(event_id=ev.id, room_id=r1.id, language_code="it", language_name="Italian")
            b2 = DBBooth(event_id=ev.id, room_id=r2.id, language_code="it", language_name="Italian")
            session.add_all([b1, b2])

            admin = await create_user(
                session, email="admin@collision.com", display_name="Admin", password_hash="hash", is_admin=True
            )
            await session.commit()
            return ev.id, r1.id, r2.id, str(r1.id), admin.id

    ev_id, r1_id, r2_id, target_ey_id, admin_id = anyio.run(_setup_colliding_rooms)

    token = create_user_token(user_id=admin_id, email="admin@collision.com", is_admin=True)
    res = client.delete(
        f"/api/events/collision-event/rooms/{target_ey_id}/booths/it?is_external=true",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 204

    async def _check_collision_db():
        async with get_session() as session:
            b2 = await session.scalar(
                select(DBBooth).where(
                    DBBooth.event_id == ev_id, DBBooth.room_id == r2_id, DBBooth.language_code == "it"
                )
            )
            assert b2 is None

            b1 = await session.scalar(
                select(DBBooth).where(
                    DBBooth.event_id == ev_id, DBBooth.room_id == r1_id, DBBooth.language_code == "it"
                )
            )
            assert b1 is not None

    anyio.run(_check_collision_db)


def test_concurrent_delete_idempotency(monkeypatch):
    """Duplicate / concurrent DELETE requests return 204 without 500 error."""
    monkeypatch.setattr(settings, "booth_access_token", "secret-test-token")
    ev_id, r_int_id, _, admin_id = anyio.run(_seed_test_event_and_booths)

    token = create_user_token(user_id=admin_id, email="admin@test.com", is_admin=True)
    headers = {"Authorization": f"Bearer {token}"}

    res1 = client.delete(f"/api/events/event-786/rooms/{r_int_id}/booths/en", headers=headers)
    assert res1.status_code == 204

    res2 = client.delete(f"/api/events/event-786/rooms/{r_int_id}/booths/en", headers=headers)
    assert res2.status_code == 204


def test_access_log_token_redactor_covers_api_paths():
    """_UvicornTokenRedactor masks token query params in /api/ routes."""
    redactor = _UvicornTokenRedactor()
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", "DELETE", "/api/events/test/rooms/1/booths/en?token=super-secret-token", "1.1", 204),
        exc_info=None,
    )
    redactor.filter(record)
    output = record.getMessage()
    assert "super-secret-token" not in output
    assert "token=[REDACTED]" in output
