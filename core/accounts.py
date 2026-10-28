"""Administrator-owned Bilibili credentials and QR login sessions.

The module deliberately has no AstrBot imports. It owns the sensitive state
and exposes small serialisable views for the WebUI; ``main.py`` remains the
only place that knows about HTTP requests, Dashboard users, and SSE responses.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast


LoginState: TypeAlias = Literal[
    "waiting", "scanned", "confirmed", "expired", "failed", "cancelled"
]
_TERMINAL_STATES = frozenset({"confirmed", "expired", "failed", "cancelled"})
_STATE_ALIASES = {"pending": "waiting", "waiting": "waiting", "scanned": "scanned"}
_VALID_STATES = _TERMINAL_STATES | {"waiting", "scanned"}
_COOKIE_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


class AccountError(RuntimeError):
    """Base error for account operations safe to surface to an administrator."""


class LoginSessionNotFound(AccountError):
    """Raised for expired, unknown, or already purged QR sessions."""


class LoginSessionAccessDenied(AccountError):
    """Raised when a Dashboard user tries to read another user's QR session."""


class CredentialStoreError(AccountError):
    """Raised when the credential file cannot be decoded or persisted."""


@dataclass(frozen=True, slots=True)
class AccountProfile:
    """Non-sensitive Bilibili account metadata displayed in the WebUI."""

    display_name: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"display_name": self.display_name}


@dataclass(frozen=True, slots=True)
class AccountCredentials:
    """Persisted Bilibili credentials; cookies are intentionally omitted from repr."""

    cookies: Mapping[str, str] = field(repr=False)
    profile: AccountProfile = field(default_factory=AccountProfile)
    saved_at: float = 0.0


@dataclass(frozen=True, slots=True)
class QrLoginStart:
    """Private result of creating a Bilibili QR login session.

    ``poll_context`` and ``qr_url`` must never be serialised directly into a
    WebUI response. The composition root turns the URL into a QR PNG first.
    """

    qr_url: str
    poll_context: object = field(repr=False)
    expires_at: float | None = None


@dataclass(frozen=True, slots=True)
class QrLoginPoll:
    """Normalized result from one Bilibili QR status poll."""

    state: str
    message: str | None = None
    cookies: Mapping[str, str] | None = field(default=None, repr=False)


StartLoginCallback: TypeAlias = Callable[[], Awaitable[QrLoginStart]]
PollLoginCallback: TypeAlias = Callable[[object], Awaitable[QrLoginPoll]]
CancelLoginCallback: TypeAlias = Callable[[object], Awaitable[None]]
ProfileCallback: TypeAlias = Callable[
    [Mapping[str, str]], Awaitable[AccountProfile | Mapping[str, Any] | None]
]


@dataclass(frozen=True, slots=True)
class BilibiliAuthenticator:
    """The narrow Bilibili login adapter supplied by the composition root."""

    start: StartLoginCallback
    poll: PollLoginCallback
    cancel: CancelLoginCallback | None = None
    profile: ProfileCallback | None = None


@dataclass(slots=True)
class _LoginSession:
    session_id: str
    owner_id: str
    poll_context: object = field(repr=False)
    qr_url: str
    expires_at: float
    created_at: float
    state: LoginState = "waiting"
    message: str | None = None
    revision: int = 0
    ended_at: float | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)


def _normalise_message(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace("\x00", "")
    return text[:240] or None


def _normalise_profile(
    value: AccountProfile | Mapping[str, Any] | None,
) -> AccountProfile:
    if isinstance(value, AccountProfile):
        display_name = value.display_name
    elif isinstance(value, Mapping):
        display_name = value.get("display_name") or value.get("nickname")
    else:
        display_name = getattr(value, "display_name", None)
    return AccountProfile(display_name=_normalise_message(display_name))


def _normalise_cookies(value: Mapping[str, str] | None) -> dict[str, str]:
    """Validate a cookie map before it can later become an HTTP header."""
    if not isinstance(value, Mapping):
        return {}
    cookies: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            continue
        name = raw_name.strip()
        cookie = raw_value.strip()
        if (
            not name
            or not cookie
            or len(name) > 256
            or len(cookie) > 8192
            or "\r" in name
            or "\n" in name
            or "\r" in cookie
            or "\n" in cookie
            or ";" in cookie
            or not _COOKIE_NAME_RE.fullmatch(name)
        ):
            continue
        cookies[name] = cookie
    return cookies


def _parse_cookie_text(value: str) -> dict[str, str]:
    """Parse a browser-style ``name=value; name2=value2`` cookie string."""
    if not isinstance(value, str):
        return {}
    pairs: dict[str, str] = {}
    for part in value.split(";"):
        name, separator, cookie = part.strip().partition("=")
        if not separator or not name or not cookie:
            continue
        pairs[name.strip()] = cookie.strip()
    return _normalise_cookies(pairs)


def _encode_credentials(record: AccountCredentials | None) -> bytes:
    payload: dict[str, object] = {"version": 1}
    if record is not None:
        payload["bilibili"] = {
            "cookies": dict(record.cookies),
            "profile": record.profile.as_dict(),
            "saved_at": record.saved_at,
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _decode_credentials(raw: bytes) -> AccountCredentials | None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CredentialStoreError("账号凭证文件无法读取") from exc
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise CredentialStoreError("账号凭证文件格式不受支持")

    raw_record = payload.get("bilibili")
    if raw_record is None:
        return None
    if not isinstance(raw_record, Mapping):
        raise CredentialStoreError("账号凭证文件格式不正确")
    cookies = _normalise_cookies(raw_record.get("cookies"))
    if not cookies:
        return None
    saved_at = raw_record.get("saved_at")
    return AccountCredentials(
        cookies=cookies,
        profile=_normalise_profile(raw_record.get("profile")),
        saved_at=float(saved_at) if isinstance(saved_at, (int, float)) else 0.0,
    )


class CredentialStore:
    """A minimal atomic JSON store for the administrator's Bilibili cookies."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def load(self) -> AccountCredentials | None:
        return await asyncio.to_thread(self._load_sync)

    async def save(self, record: AccountCredentials | None) -> None:
        await asyncio.to_thread(self._write_sync, _encode_credentials(record))

    def _load_sync(self) -> AccountCredentials | None:
        if not self.path.exists():
            return None
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise CredentialStoreError("账号凭证文件无法读取") from exc
        return _decode_credentials(raw)

    def _write_sync(self, payload: bytes) -> None:
        temporary_path: Path | None = None
        file_descriptor: int | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            temporary_path = Path(temporary_name)
            if os.name != "nt":
                os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "wb") as file:
                file_descriptor = None
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise CredentialStoreError("账号凭证文件无法保存") from exc
        finally:
            if file_descriptor is not None:
                try:
                    os.close(file_descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


class AccountService:
    """Coordinates one administrator-owned Bilibili account and QR session."""

    def __init__(
        self,
        store: CredentialStore,
        authenticator: BilibiliAuthenticator,
        *,
        poll_interval: float = 2.0,
        fallback_session_ttl: float = 180.0,
        terminal_session_ttl: float = 300.0,
    ) -> None:
        if poll_interval <= 0 or fallback_session_ttl <= 0 or terminal_session_ttl <= 0:
            raise ValueError("account service durations must be positive")

        self._store = store
        self._authenticator = authenticator
        self._poll_interval = poll_interval
        self._fallback_session_ttl = fallback_session_ttl
        self._terminal_session_ttl = terminal_session_ttl
        self._credentials: AccountCredentials | None = None
        self._sessions: dict[str, _LoginSession] = {}
        self._active_session_id: str | None = None
        self._loaded = False
        self._storage_error: str | None = None
        self._lock = asyncio.Lock()

    async def restore_credentials(self) -> None:
        """Load persisted credentials once during plugin startup.

        A malformed credential file leaves the plugin in anonymous mode while
        preserving the file for administrator inspection.
        """
        async with self._lock:
            if self._loaded:
                return
            try:
                self._credentials = await self._store.load()
            except CredentialStoreError:
                self._credentials = None
                self._storage_error = "账号凭证文件不可读取"
            self._loaded = True

    def cookies(self) -> dict[str, str]:
        """Return a defensive credential copy for the Bilibili HTTP client."""
        record = self._credentials
        return dict(record.cookies) if record else {}

    def has_credentials(self) -> bool:
        return bool(self.cookies())

    async def status_payload(self) -> dict[str, Any]:
        """Return the only account view suitable for WebUI serialization."""
        await self.restore_credentials()
        async with self._lock:
            self._purge_sessions_locked(time.time())
            profile = (
                self._credentials.profile if self._credentials else AccountProfile()
            )
            return {
                "account": {
                    "state": "connected" if self._credentials else "anonymous",
                    "display_name": profile.display_name,
                },
                "storage": {"state": "error" if self._storage_error else "ready"},
            }

    async def start_login(self, owner_id: str) -> dict[str, Any]:
        """Create the sole QR session and start its server-side polling task."""
        await self.restore_credentials()
        owner = owner_id.strip()
        if not owner:
            raise LoginSessionAccessDenied("未识别到 Dashboard 管理员")

        async with self._lock:
            await self._cancel_active_locked("二维码登录已被新的请求替换")
            self._purge_sessions_locked(time.time(), discard_terminal=True)
            try:
                started = await self._authenticator.start()
            except Exception as exc:
                raise AccountError("无法创建登录二维码") from exc
            if not isinstance(started, QrLoginStart) or not started.qr_url.strip():
                raise AccountError("Bilibili 未返回有效的登录二维码")

            now = time.time()
            expires_at = started.expires_at or now + self._fallback_session_ttl
            if expires_at <= now:
                expires_at = now + self._fallback_session_ttl
            session = _LoginSession(
                session_id=secrets.token_urlsafe(24),
                owner_id=owner,
                poll_context=started.poll_context,
                qr_url=started.qr_url.strip(),
                expires_at=expires_at,
                created_at=now,
            )
            self._sessions[session.session_id] = session
            self._active_session_id = session.session_id
            session.task = asyncio.create_task(
                self._poll_login(session), name="bili-player-bilibili-qr-login"
            )
            return self._session_payload(session)

    async def cancel_login(self, session_id: str, owner_id: str) -> None:
        await self.restore_credentials()
        async with self._lock:
            await self._cancel_session_locked(
                self._owned_session_locked(session_id, owner_id), "二维码登录已取消"
            )

    async def logout(self) -> None:
        """Discard local cookies; this deliberately does not revoke remote sessions."""
        await self.restore_credentials()
        async with self._lock:
            await self._cancel_active_locked("账号已退出")
            try:
                await self._store.save(None)
            except CredentialStoreError as exc:
                raise AccountError("无法保存退出状态") from exc
            self._credentials = None

    async def import_credentials(self, cookie_text: str) -> AccountProfile:
        """Validate and persist manually supplied administrator cookies.

        The cookies must belong to a logged-in Bilibili session; otherwise the
        import is rejected so a stale clipboard value can never silently
        disable restricted-content playback.
        """
        await self.restore_credentials()
        cookies = _parse_cookie_text(cookie_text)
        if not cookies:
            raise AccountError("未识别到有效的 Cookie 字段")
        profile = await self._load_profile(cookies)
        if not profile.display_name:
            raise AccountError("Cookie 无效或已失效，无法登录 Bilibili")
        async with self._lock:
            await self._cancel_active_locked("账号凭证已手动更新")
            record = AccountCredentials(
                cookies=cookies,
                profile=profile,
                saved_at=time.time(),
            )
            try:
                await self._store.save(record)
            except CredentialStoreError as exc:
                raise AccountError("无法保存登录状态") from exc
            self._credentials = record
        return profile

    async def login_events(
        self, session_id: str, owner_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield initial, changed, and keepalive snapshots for an SSE handler."""
        await self.restore_credentials()
        async with self._lock:
            session = self._owned_session_locked(session_id, owner_id)

        revision = -1
        while True:
            async with session.changed:
                if (
                    session.revision == revision
                    and session.state not in _TERMINAL_STATES
                ):
                    try:
                        await asyncio.wait_for(session.changed.wait(), timeout=20.0)
                    except TimeoutError:
                        pass
                payload = self._session_payload(session)
                revision = session.revision
                terminal = session.state in _TERMINAL_STATES
            yield payload
            if terminal:
                return

    async def aclose(self) -> None:
        """Stop the transient QR polling task when AstrBot unloads the plugin."""
        async with self._lock:
            await self._cancel_active_locked("插件已停止")

    async def _poll_login(self, session: _LoginSession) -> None:
        consecutive_failures = 0
        try:
            while True:
                if time.time() >= session.expires_at:
                    await self._set_session_state(session, "expired", "二维码已过期")
                    return

                try:
                    result = await self._authenticator.poll(session.poll_context)
                    state = self._normalise_login_state(result.state)
                    consecutive_failures = 0
                except asyncio.CancelledError:
                    raise
                except Exception:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        await self._set_session_state(
                            session, "failed", "无法检查二维码登录状态"
                        )
                        return
                    await self._set_session_state(
                        session, session.state, "正在重试登录状态检查"
                    )
                    await asyncio.sleep(self._poll_interval)
                    continue

                if state == "confirmed":
                    cookies = _normalise_cookies(result.cookies)
                    if not cookies:
                        await self._set_session_state(
                            session, "failed", "登录未返回有效凭证"
                        )
                        return
                    record = AccountCredentials(
                        cookies=cookies,
                        profile=await self._load_profile(cookies),
                        saved_at=time.time(),
                    )
                    try:
                        async with self._lock:
                            if (
                                self._active_session_id != session.session_id
                                or session.state in _TERMINAL_STATES
                            ):
                                return
                            await self._store.save(record)
                            self._credentials = record
                    except CredentialStoreError:
                        await self._set_session_state(
                            session, "failed", "无法保存登录状态"
                        )
                        return
                    await self._set_session_state(
                        session, "confirmed", result.message or "登录成功"
                    )
                    return

                await self._set_session_state(session, state, result.message)
                if state in _TERMINAL_STATES:
                    return
                await asyncio.sleep(self._poll_interval)
        finally:
            async with self._lock:
                if self._active_session_id == session.session_id:
                    self._active_session_id = None

    async def _load_profile(self, cookies: Mapping[str, str]) -> AccountProfile:
        if self._authenticator.profile is None:
            return AccountProfile()
        try:
            return _normalise_profile(await self._authenticator.profile(cookies))
        except Exception:
            return AccountProfile()

    async def _cancel_active_locked(self, message: str) -> None:
        session = (
            self._sessions.get(self._active_session_id)
            if self._active_session_id is not None
            else None
        )
        if session is not None:
            await self._cancel_session_locked(session, message)

    async def _cancel_session_locked(
        self, session: _LoginSession, message: str
    ) -> None:
        if session.state in _TERMINAL_STATES:
            return
        task = session.task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        if self._authenticator.cancel is not None:
            try:
                await self._authenticator.cancel(session.poll_context)
            except Exception:
                pass
        await self._set_session_state(session, "cancelled", message)
        if self._active_session_id == session.session_id:
            self._active_session_id = None

    async def _set_session_state(
        self, session: _LoginSession, state: LoginState, message: str | None
    ) -> None:
        async with session.changed:
            session.state = state
            session.message = _normalise_message(message)
            if state in _TERMINAL_STATES and session.ended_at is None:
                session.ended_at = time.time()
            session.revision += 1
            session.changed.notify_all()

    def _owned_session_locked(self, session_id: str, owner_id: str) -> _LoginSession:
        self._purge_sessions_locked(time.time())
        session = self._sessions.get(session_id)
        if session is None:
            raise LoginSessionNotFound("二维码登录会话不存在或已过期")
        if not owner_id or not secrets.compare_digest(session.owner_id, owner_id):
            raise LoginSessionAccessDenied("无权访问该二维码登录会话")
        return session

    def _purge_sessions_locked(
        self, now: float, *, discard_terminal: bool = False
    ) -> None:
        for session_id, session in tuple(self._sessions.items()):
            terminal_since = session.ended_at or max(
                session.created_at, session.expires_at
            )
            if session.state in _TERMINAL_STATES and (
                discard_terminal or now >= terminal_since + self._terminal_session_ttl
            ):
                self._sessions.pop(session_id, None)

    @staticmethod
    def _normalise_login_state(raw_state: str) -> LoginState:
        state = _STATE_ALIASES.get(
            str(raw_state).strip().lower(), str(raw_state).strip().lower()
        )
        return cast(LoginState, state if state in _VALID_STATES else "failed")

    @staticmethod
    def _session_payload(session: _LoginSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "state": session.state,
            "message": session.message,
            "qr_url": session.qr_url,
            "expires_at": session.expires_at,
        }


__all__ = [
    "AccountError",
    "AccountProfile",
    "AccountService",
    "BilibiliAuthenticator",
    "CredentialStore",
    "CredentialStoreError",
    "LoginSessionAccessDenied",
    "LoginSessionNotFound",
    "QrLoginPoll",
    "QrLoginStart",
]
