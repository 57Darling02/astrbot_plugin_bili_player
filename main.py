"""AstrBot entry point for the intentionally small bili-player plugin."""

import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import CustomFilter
from astrbot.api.message_components import File, Record, Video
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import error_response, json_response, request, stream_response
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.session_waiter import (
    FILTERS,
    SessionController,
    SessionFilter,
    SessionWaiter,
)

from .core.accounts import (
    AccountError,
    AccountService,
    BilibiliAuthenticator,
    CredentialStore,
    LoginSessionAccessDenied,
    LoginSessionNotFound,
    QrLoginPoll,
    QrLoginStart,
)
from .core.bilibili import (
    BilibiliClient,
    parse_bilibili_video_ref,
    parse_bilibili_video_refs,
)
from .core.media import (
    FfmpegUnavailableError,
    MediaError,
    MediaLimits,
    MediaStore,
)
from .core.models import BilibiliCandidate, LocalMedia, SearchSnapshot
from .core.selection import SearchSnapshotStore
from .core.settings import PluginLimits, PluginSettings
from .core.services import (
    SEARCH_LIMIT,
    DeliveryError,
    DeliveryResult,
    DeliveryService,
    MusicSearchError,
    SearchService,
    format_search_results,
    summarize_search_candidates,
)


PLUGIN_NAME = "astrbot_plugin_bili_player"
INTERACTION_TIMEOUT_SECONDS = 90
SEARCH_SNAPSHOT_TTL_SECONDS = 300.0
SEARCH_SNAPSHOT_MAX_ENTRIES = 1024
# AstrBot's weixin_oc adapter accepts File outbound but ignores Record.
_VOICE_AS_FILE_PLATFORMS = frozenset({"weixin_oc"})
_CANONICAL_RECORDING_PREFERENCES = frozenset({"原版", "原唱", "original"})
_DELIVERY_KEYS = frozenset({"auto", "video", "audio", "download"})


class _StaleLlmDelivery(Exception):
    """An obsolete hidden LLM delivery that must not message the user."""


_CHINESE_SELECTION_POSITIONS = tuple("一二三四五六七八九十")
if SEARCH_LIMIT > len(_CHINESE_SELECTION_POSITIONS):
    raise RuntimeError("selection grammar needs more Chinese position names")
_SELECTION_POSITION_MAP = {
    **{str(position): position for position in range(1, SEARCH_LIMIT + 1)},
    **{
        character: position
        for position, character in enumerate(
            _CHINESE_SELECTION_POSITIONS[:SEARCH_LIMIT], start=1
        )
    },
}
_SELECTION_POSITION_PATTERN = "|".join(
    sorted(
        (re.escape(value) for value in _SELECTION_POSITION_MAP), key=len, reverse=True
    )
)
_SELECTION_RE = re.compile(
    r"^(?:(?:我|我要|我想|帮我|请)\s*)?"
    r"(?:(音频下载|下载|听(?:歌)?|播放|视频|音频)\s*)?"
    r"(?:(?:选择|选)\s*)?"
    rf"(?:第\s*)?({_SELECTION_POSITION_PATTERN})\s*(?:首(?:歌)?|个|号)?"
    r"(?:\s*(音频下载|下载|听(?:歌)?|播放|视频|音频))?"
    r"(?:[，,。！？!]\s*)*$"
)


class _DeliveryMode(str, Enum):
    """The three AstrBot transport forms for an already-prepared media file."""

    VOICE = "voice"
    DOWNLOAD = "download"
    VIDEO = "video"


def _media_limits_for(action: _DeliveryMode, limits: PluginLimits) -> MediaLimits:
    """Keep user-visible delivery intent aligned with one configured budget."""

    if action is _DeliveryMode.DOWNLOAD:
        return limits.download
    if action is _DeliveryMode.VIDEO:
        return limits.video
    return limits.voice


@dataclass(frozen=True, slots=True)
class _BilibiliRequest:
    """The structured song identity supplied by one LLM tool call."""

    title: str
    artist: str | None = None
    version: str | None = None

    @classmethod
    def from_fields(
        cls,
        title: object,
        *,
        artist: object | None = None,
        version: object | None = None,
    ) -> "_BilibiliRequest":
        normalized_title = _normalize_music_request_field(title)
        if not normalized_title:
            raise ValueError("请提供歌曲名称")
        normalized_artist = _normalize_music_request_field(artist)
        normalized_version = _normalize_music_request_field(version)
        return cls(
            title=normalized_title,
            artist=normalized_artist or None,
            version=normalized_version or None,
        )

    @property
    def query(self) -> str:
        """Keep source-query construction in the plugin, not the LLM."""

        parts = [self.title]
        if self.artist:
            parts.append(self.artist)
        if self.version and not self.prefers_canonical_recording:
            parts.append(self.version)
        return " ".join(parts)

    @property
    def prefers_canonical_recording(self) -> bool:
        return bool(self.version and _is_canonical_recording_preference(self.version))


def _normalize_music_request_field(value: object | None) -> str:
    return " ".join(("" if value is None else str(value)).replace("\x00", "").split())


def _is_canonical_recording_preference(version: str) -> bool:
    return "".join(version.casefold().split()) in _CANONICAL_RECORDING_PREFERENCES


def _tool_error(message: str) -> str:
    """Return a compact structured failure to the LLM without chat side effects."""

    return json.dumps(
        {"status": "error", "message": message},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _llm_candidate_result(snapshot: SearchSnapshot) -> str:
    """Serialize only the selection evidence the model needs for one choice."""

    return json.dumps(
        {
            "status": "candidates",
            "search_id": snapshot.search_id,
            "requires_user_choice": _requires_user_page_choice(snapshot),
            "candidates": [
                {
                    "position": candidate.position,
                    "title": candidate.title,
                    "duration": candidate.duration,
                    "search_title": candidate.search_title,
                    "page_title": candidate.page_title,
                }
                for candidate in summarize_search_candidates(snapshot)
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_candidate_position(value: object) -> int:
    """Accept only a JSON integer inside the active candidate range."""

    if isinstance(value, int) and not isinstance(value, bool):
        position = value
    elif isinstance(value, str) and value.strip().isdigit():
        position = int(value.strip())
    else:
        raise ValueError("候选序号无效，请重新搜索")
    if not 1 <= position <= SEARCH_LIMIT:
        raise ValueError("候选序号无效，请重新搜索")
    return position


@dataclass(slots=True)
class _SelectionWait:
    """The active host waiter for one chat while a user chooses a result."""

    controller: SessionController
    finished: asyncio.Event
    task: asyncio.Task[None] | None = None
    cancelled: bool = False


@dataclass(slots=True)
class _LlmSearch:
    """A cancellable, short-lived authorization for one hidden candidate set."""

    expires_at: float
    search_id: str | None = None
    delivery: str = "video"


class _SelectionSessionFilter(SessionFilter):
    """Route only an active result-selection reply into AstrBot's waiter."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def filter(self, event: AstrMessageEvent) -> str:
        if event.unified_msg_origin != self._session_id:
            return ""
        message = " ".join(event.message_str.split())
        if _is_selection_reply(message):
            return self._session_id
        return ""


class _DiscardSelectionWaitFilter(CustomFilter):
    """Activate only for a new message that supersedes an active selection."""

    def filter(self, event: AstrMessageEvent, _config: Any) -> bool:
        message = " ".join(event.message_str.split())
        if _is_selection_reply(message):
            return False
        return any(
            isinstance(session_filter, _SelectionSessionFilter)
            and session_filter.session_id == event.unified_msg_origin
            for session_filter in tuple(FILTERS)
        )


def _bilibili_request_parameters() -> dict[str, Any]:
    """Build the shared, small structured-intent contract for LLM tools."""

    return {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "作品名；“唱首歌”时先选定一首具体歌曲。",
            },
            "artist": {
                "type": "string",
                "description": "歌手、乐队或 UP 主；无明确线索时省略。",
            },
            "version": {
                "type": "string",
                "description": "用户明确指定的版本偏好；未指定时省略。",
            },
        },
        "required": ["title"],
        "additionalProperties": False,
    }


def _tool_event(context: Any) -> AstrMessageEvent | None:
    event = getattr(getattr(context, "context", None), "event", None)
    return event if isinstance(event, AstrMessageEvent) else None


class FindInBilibiliTool(FunctionTool):
    """Unified pre-step search for listening, watching, or downloading."""

    def __init__(self, plugin: "ListenMusicPlugin") -> None:
        super().__init__(
            name="find_in_bilibili",
            description=(
                "本插件是 Bilibili 视频/音频输出工具。当用户想通过聊天看视频、听歌、点歌、播放某首歌或下载音频时，必须先调用本工具；不要自己发送链接或仅用文字描述媒体。"
                "适用示例：我要看晴天、我想听周杰伦、来一首、唱首歌、下载这首歌。title 必须是纯作品名——去掉“我要看/我要听/播放”等指令词，也不要附加“原版/无损”等偏好词；可选传歌手和版本偏好。"
                "delivery：auto=用户未明确看/听/下载（如只说“播放/来一首”），交给插件配置；video=明确要看/视频；audio=明确要听/唱/音频；download=下载音频，后续必须让用户选择。"
                "本工具只搜索候选，不向用户发送内容；成功后调用 deliver_media 完成实际发送。只能使用返回的 search_id 和 position。"
                "候选评估：作品名精确或完整匹配优先，歌手线索佐证，必须遵守版本偏好；Live/翻唱/AI/DJ/伴奏/MV 等标签仅作证据。"
                "没有可信候选时不要猜测或发送，简短请求用户补充作品名。本工具失败时只返回 error 信息，由你用自然语言向用户转述，不要直接输出 JSON。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    **_bilibili_request_parameters()["properties"],
                    "delivery": {
                        "type": "string",
                        "enum": ["auto", "video", "audio", "download"],
                        "description": "auto 仅在用户确实没有明确媒体意图时使用。",
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        )
        self._plugin = plugin

    async def call(self, context: Any, **kwargs: Any) -> str:
        event = _tool_event(context)
        if event is None:
            return _tool_error("无法获取当前聊天会话。")
        return await self._plugin.find_in_bilibili_for_llm(
            event,
            kwargs.get("title", ""),
            artist=kwargs.get("artist"),
            version=kwargs.get("version"),
            delivery=kwargs.get("delivery"),
        )


class DeliverMediaTool(FunctionTool):
    """Deliver a chosen candidate, automatically or via a user-owned pick."""

    def __init__(self, plugin: "ListenMusicPlugin") -> None:
        reply_rule = (
            "插件会在下载、转码前主动发送一条准备提示；你不要自行输出预告、过程或确认文字。"
            "直接交付成功后本工具不返回内容，AstrBot 会直接结束本轮；"
            "失败时返回 error，由你用自然语言向用户转述，不要输出 JSON。"
        )
        super().__init__(
            name="deliver_media",
            description=(
                "本工具把 find_in_bilibili 选中的 Bilibili 候选实际发送为视频、语音或文件。不要绕过它自行发送链接、标题或内容简介。"
                "只能在 find_in_bilibili 成功后调用，并使用其返回的 search_id 和 position。"
                "交付媒体类型由 find_in_bilibili 的 delivery 和插件配置决定，本工具不再接收 delivery。"
                "直接交付（let_user_choose=false）：自动发送；视频请求固定发第一个候选，精确 AV/BV 多分 P 时展示候选。"
                + reply_rule
                + "下载音频或让用户挑（let_user_choose=true）：展示候选，用户回“序号”发视频、“序号 音频”播放、“序号 音频下载”下载音频文件；"
                "下载音频必须让用户选。不得编造 search_id/序号，不输出过程、解释或确认文字。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "search_id": {
                        "type": "string",
                        "description": "find_in_bilibili 返回的当前会话 search_id。",
                    },
                    "position": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": SEARCH_LIMIT,
                        "description": "find_in_bilibili 返回的候选序号；let_user_choose 为 true 时可填 1。",
                    },
                    "let_user_choose": {
                        "type": "boolean",
                        "description": "下载音频或让用户挑时 true，展示候选；直接听/看时省略。",
                    },
                },
                "required": ["search_id", "position"],
                "additionalProperties": False,
            },
        )
        self._plugin = plugin

    async def call(self, context: Any, **kwargs: Any) -> str | None:
        event = _tool_event(context)
        if event is None:
            return None
        return await self._plugin.deliver_media_for_llm(
            event,
            kwargs.get("search_id", ""),
            kwargs.get("position"),
            let_user_choose=bool(kwargs.get("let_user_choose", False)),
        )


class ListenMusicPlugin(Star):
    """Bilibili 单源搜索、语音听歌和文件下载。"""

    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context, config)
        self._settings = PluginSettings.from_mapping(config)
        self._http: aiohttp.ClientSession | None = None
        self._accounts: AccountService | None = None
        self._bilibili: BilibiliClient | None = None
        self._media: MediaStore | None = None
        self._search: SearchService | None = None
        self._delivery: DeliveryService | None = None
        self._selection_waits: dict[str, _SelectionWait] = {}
        self._llm_searches: dict[str, _LlmSearch] = {}
        self._selection_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        connector = aiohttp.TCPConnector(limit=16, limit_per_host=8, ttl_dns_cache=300)
        http = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=180, connect=15, sock_read=90),
        )
        try:
            accounts = AccountService(
                CredentialStore(data_dir / "accounts.json"),
                BilibiliAuthenticator(
                    start=self._start_bilibili_login,
                    poll=self._poll_bilibili_login,
                    cancel=self._cancel_bilibili_login,
                    profile=self._bilibili_profile,
                ),
            )
            await accounts.restore_credentials()
            bilibili = BilibiliClient(
                http,
                credentials_getter=accounts.cookies,
            )
            media = MediaStore(http, data_dir / "media")
            await media.reclaim_stale()
            search = SearchService(
                bilibili,
                SearchSnapshotStore(
                    ttl_seconds=SEARCH_SNAPSHOT_TTL_SECONDS,
                    max_entries=SEARCH_SNAPSHOT_MAX_ENTRIES,
                ),
            )

            self._http = http
            self._accounts = accounts
            self._bilibili = bilibili
            self._media = media
            self._search = search
            self._delivery = DeliveryService(
                bilibili=bilibili,
                media=media,
            )
            self._register_account_routes()
            self.context.add_llm_tools(
                FindInBilibiliTool(self),
                DeliverMediaTool(self),
            )
            self._initialized = True
            logger.info("bili-player plugin initialized")
        except Exception:
            await http.close()
            raise

    async def terminate(self) -> None:
        media, accounts, http = self._media, self._accounts, self._http
        self._initialized = False
        self._unregister_account_routes()
        async with self._selection_lock:
            selection_waits = tuple(self._selection_waits.values())
            self._selection_waits.clear()
            self._llm_searches.clear()
        for selection in selection_waits:
            selection.cancelled = True
            selection.controller.stop()
        selection_tasks = tuple(
            selection.task
            for selection in selection_waits
            if selection.task is not None
            and selection.task is not asyncio.current_task()
        )
        if selection_tasks:
            await asyncio.gather(*selection_tasks, return_exceptions=True)
        self._delivery = None
        self._search = None
        self._media = None
        self._accounts = None
        self._bilibili = None
        self._http = None
        if media is not None:
            await media.aclose()
        if accounts is not None:
            await accounts.aclose()
        if http is not None and not http.closed:
            await http.close()

    @filter.command("搜索歌曲")
    async def search_song(self, event: AstrMessageEvent, query: GreedyStr):
        """搜索歌曲 <关键词>"""
        async for result in self._run_search_command(event, query):
            yield result

    @filter.command("搜索视频")
    async def search_video(self, event: AstrMessageEvent, query: GreedyStr):
        """搜索视频 <关键词>"""
        if not self._settings.video_allowed:
            yield event.plain_result(
                "当前插件配置未开启视频交付。\n可在插件配置中调整“默认媒体类型”。"
            )
            event.stop_event()
            return
        async for result in self._run_search_command(
            event, query, usage="搜索视频 <关键词>"
        ):
            yield result

    @filter.command("/我要听")
    async def listen_command(self, event: AstrMessageEvent, query: GreedyStr):
        """兜底命令：/我要听 <作品名> 展示候选，由用户选择。"""
        if not self._settings.audio_allowed:
            yield event.plain_result(
                "当前插件配置未开启音频交付。\n可在插件配置中调整“默认媒体类型”。"
            )
            event.stop_event()
            return
        async for result in self._run_search_command(
            event, query, usage="/我要听 <作品名>"
        ):
            yield result

    @filter.command("/我要看")
    async def watch_command(self, event: AstrMessageEvent, query: GreedyStr):
        """兜底命令：/我要看 <作品名> 直接发第一个视频；含 AV/BV 时先展示分 P。"""
        event.should_call_llm(True)
        if not self._settings.video_allowed:
            yield event.plain_result(
                "当前插件配置未开启视频交付。\n可在插件配置中调整“默认媒体类型”。"
            )
            event.stop_event()
            return
        query = query.strip()
        if not query:
            yield event.plain_result("请使用：/我要看 <作品名>")
            event.stop_event()
            return
        if parse_bilibili_video_ref(query) is not None:
            async for result in self._run_search_command(
                event, query, usage="/我要看 <作品名>"
            ):
                yield result
            return
        session_id = event.unified_msg_origin
        self._clear_llm_search(session_id)
        await self._cancel_selection_wait(session_id)
        try:
            snapshot = await self._require_search().search(
                session_id=session_id,
                query=query,
                video_ref=None,
            )
        except MusicSearchError as exc:
            yield event.plain_result(str(exc))
            event.stop_event()
            return
        except Exception:
            logger.exception("bili-player watch command failed")
            yield event.plain_result("搜索视频时发生错误，请稍后重试")
            event.stop_event()
            return
        candidate = snapshot.candidate_at(1)
        if candidate is None:
            yield event.plain_result("没有找到可发送的视频。")
            event.stop_event()
            return
        try:
            await self._deliver_media(
                event,
                candidate=candidate,
                action=_DeliveryMode.VIDEO,
            )
        except (DeliveryError, FfmpegUnavailableError, MediaError) as exc:
            yield event.plain_result(str(exc))
            event.stop_event()
            return
        except Exception:
            logger.exception("bili-player watch command delivery failed")
            yield event.plain_result("视频发送失败，请稍后重试。")
            event.stop_event()
            return
        event.stop_event()

    async def _run_search_command(
        self,
        event: AstrMessageEvent,
        query: GreedyStr,
        *,
        usage: str = "搜索歌曲 <关键词>",
    ) -> AsyncIterator[str]:
        """Show the candidate catalogue and own the following selection."""

        event.should_call_llm(True)
        query = query.strip()
        if not query:
            yield event.plain_result(f"请使用：{usage}")
            event.stop_event()
            return
        session_id = event.unified_msg_origin
        self._clear_llm_search(session_id)
        await self._cancel_selection_wait(session_id)
        fuzzy_query = len(parse_bilibili_video_refs(query)) > 1
        try:
            snapshot = await self._require_search().search(
                session_id=session_id,
                query=query,
                video_ref=None if fuzzy_query else parse_bilibili_video_ref(query),
                fuzzy_query=fuzzy_query,
            )
        except MusicSearchError as exc:
            yield event.plain_result(str(exc))
            event.stop_event()
            return
        except Exception:
            logger.exception("bili-player command search failed")
            yield event.plain_result(f"{_command_error_label(usage)}，请稍后重试")
            event.stop_event()
            return

        if not await self._start_selection_wait(event, snapshot):
            yield event.plain_result("插件正在停止，无法继续选歌。")
            event.stop_event()
            return
        yield event.plain_result(self._format_selection_results(snapshot))
        event.stop_event()

    async def find_in_bilibili_for_llm(
        self,
        event: AstrMessageEvent,
        title: object,
        *,
        artist: object | None = None,
        version: object | None = None,
        delivery: object | None = None,
    ) -> str:
        """Create a one-shot candidate snapshot for LLM-side direct selection."""
        session_id = event.unified_msg_origin
        settings = getattr(self, "_settings", None) or PluginSettings()
        lease: _LlmSearch | None = None
        try:
            message_refs = parse_bilibili_video_refs(event.message_str)
            fuzzy_query = len(message_refs) > 1
            lease = self._begin_llm_search(session_id)
            await self._cancel_selection_wait(session_id)
            if not self._is_current_llm_search(session_id, lease):
                return _tool_error("歌曲请求已被新的消息替换")
            music = _BilibiliRequest.from_fields(
                title,
                artist=artist,
                version=version,
            )
            delivery_key = _resolve_delivery_key(delivery, settings)
            search_kwargs: dict[str, object] = {
                "session_id": session_id,
                "query": music.query,
                "song_title": None,
                "fuzzy_query": False,
            }
            if fuzzy_query:
                # 防抖插件把多个命令拼成了一段文本：不猜任务边界，
                # 整段作为模糊关键词交给 Bilibili，并强制用户自行选择。
                search_kwargs["query"] = event.message_str.strip()
                search_kwargs["fuzzy_query"] = True
            else:
                # 请求本身是 AV/BV 号时走精确详情（关键词搜索对 BV 召回不可靠）；
                # 普通作品名才走关键词。语义提取仍由前置 LLM 负责。
                video_ref = parse_bilibili_video_ref(
                    music.title
                ) or parse_bilibili_video_ref(event.message_str)
                if video_ref is not None:
                    # 精确视频保留全部分 P，由用户选择。
                    search_kwargs["video_ref"] = video_ref
                elif delivery_key in {"audio", "download"}:
                    # 只有可信的普通音频意图才做歌名/分P语义筛选。
                    search_kwargs["song_title"] = music.title
            snapshot = await self._require_search().search(**search_kwargs)

            if not self._complete_llm_search(session_id, lease, snapshot, delivery_key):
                return _tool_error("歌曲请求已被新的消息替换")
            return _llm_candidate_result(snapshot)
        except asyncio.CancelledError:
            if lease is not None:
                self._discard_llm_search(session_id, lease)
            raise
        except (MusicSearchError, ValueError) as exc:
            if lease is not None:
                self._discard_llm_search(session_id, lease)
            return _tool_error(str(exc))
        except Exception:
            if lease is not None:
                self._discard_llm_search(session_id, lease)
            logger.exception("bili-player LLM candidate search failed")
            return _tool_error("搜索歌曲时发生错误，请稍后重试。")

    async def deliver_media_for_llm(
        self,
        event: AstrMessageEvent,
        search_id: object,
        position: object,
        *,
        let_user_choose: bool = False,
    ) -> str | None:
        """Terminally deliver one candidate from the active LLM search snapshot.

        The media type was fixed by ``find_in_bilibili`` and the plugin
        settings. A successful delivery returns ``None`` so AstrBot ends the
        agent loop without asking the model for another chat message.
        """
        session_id = event.unified_msg_origin
        settings = getattr(self, "_settings", None) or PluginSettings()
        try:
            normalized_search_id = str(search_id).strip()
            requested_position = _parse_candidate_position(position)
            selected_position = requested_position
            lease = self._active_llm_search(session_id, normalized_search_id)
            if lease is None:
                raise _StaleLlmDelivery("search lease missing or superseded")

            snapshot = self._require_search().snapshot(
                search_id=normalized_search_id,
                session_id=session_id,
            )
            if snapshot is None:
                self._discard_llm_search(session_id, lease)
                raise _StaleLlmDelivery("search snapshot missing or expired")
            delivery_key = lease.delivery
            requested_candidate = snapshot.candidate_at(requested_position)
            if requested_candidate is None:
                raise DeliveryError("候选无效或已过期，请重新搜索")
            if _requires_user_page_choice(snapshot):
                # An exact AV/BV reference exposes every page; only the user
                # may pick the concrete page for a multi-page video.
                let_user_choose = True
            elif not let_user_choose and delivery_key == "video":
                # Direct video requests use Bilibili's first result; video has
                # no music-semantic ranking step for the LLM to reproduce.
                selected_position = 1
            candidate = snapshot.candidate_at(selected_position)
            if candidate is None:
                raise DeliveryError("候选无效或已过期，请重新搜索")
            delivery_action = _action_for_delivery_key(delivery_key, settings)
            if delivery_key == "download":
                # Downloading an audio file is never automatic: the user must
                # confirm the exact candidate from the visible catalogue.
                let_user_choose = True
            if bool(let_user_choose):
                if not await self._start_selection_wait(event, snapshot):
                    self._discard_llm_search(session_id, lease)
                    raise DeliveryError("插件正在停止，无法继续选歌")
                await event.send(
                    event.plain_result(self._format_selection_results(snapshot))
                )
                # AstrBot ends the agent loop only for a None tool result.
                # Safe here: event.send() either succeeded or raised into the
                # error path below.
                return None
            if _selection_requires_download(
                candidate, delivery_action, settings.limits.voice
            ):
                # Voice messages cannot carry such a long track; deliver the
                # audio file instead so the listen request still completes.
                delivery_action = _DeliveryMode.DOWNLOAD
            await self._deliver_media(
                event,
                candidate=candidate,
                action=delivery_action,
            )
            # Consume the lease only after delivery succeeded; a failed
            # delivery keeps the snapshot so the model can retry once.
            self._consume_llm_search(session_id, lease)
            # None is the only AstrBot signal for "already sent; end loop".
            # The plugin already sent preparation feedback before download, and
            # no closing line is sent after media delivery. All failure paths
            # return an error string instead.
            return None
        except _StaleLlmDelivery as exc:
            # The originating user request has already moved on.  Do not leak
            # an internal lease race as a confusing chat message.
            logger.info("bili-player ignored stale LLM delivery: %s", exc)
            return _tool_error("候选已失效，请重新搜索")
        except (
            ValueError,
            DeliveryError,
            FfmpegUnavailableError,
            MediaError,
        ) as exc:
            return _tool_error(str(exc))
        except Exception:
            logger.exception("bili-player LLM candidate delivery failed")
            return _tool_error("歌曲发送失败，请稍后重试。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=11)
    async def discard_llm_search_on_new_message(self, event: AstrMessageEvent) -> None:
        """A new chat message must not revive a prior hidden model selection."""

        self._clear_llm_search(event.unified_msg_origin)

    @filter.custom_filter(_DiscardSelectionWaitFilter)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def discard_selection_wait_on_new_message(
        self, event: AstrMessageEvent
    ) -> None:
        """Discard an interrupted manual selection without claiming the message."""

        if _is_selection_reply(" ".join(event.message_str.split())):
            return
        await self._cancel_selection_wait(event.unified_msg_origin)
        self._clear_llm_search(event.unified_msg_origin)

    async def _deliver_media(
        self,
        event: AstrMessageEvent,
        *,
        candidate: BilibiliCandidate,
        action: _DeliveryMode,
    ) -> None:
        """Send preparation feedback, then materialize and deliver one item."""

        await event.send(event.plain_result(_delivery_preparation_message(action)))
        preparation = asyncio.create_task(
            self._deliver_candidate(candidate, action),
            name=f"bili-player-delivery-{candidate.candidate_id}",
        )
        try:
            prepared = await preparation
        except BaseException:
            await self._discard_delivery_preparation(preparation)
            raise
        await self._send_delivery(event, prepared, action)

    async def _discard_delivery_preparation(
        self, preparation: asyncio.Task[DeliveryResult]
    ) -> None:
        """Cancel or release a prepared item when it never reaches the sender."""

        if not preparation.done():
            preparation.cancel()
        try:
            prepared = await preparation
        except asyncio.CancelledError:
            return
        except Exception:
            return
        await self._require_media().release(prepared.media)

    async def _start_selection_wait(
        self, event: AstrMessageEvent, snapshot: SearchSnapshot
    ) -> bool:
        """Register the only selection path for a search result in this chat."""

        session_id = event.unified_msg_origin
        selection_filter = _SelectionSessionFilter(session_id)
        waiter = SessionWaiter(selection_filter, session_id, False)
        selection = _SelectionWait(waiter.session_controller, asyncio.Event())
        if not await self._replace_selection_wait(session_id, selection):
            return False

        # SessionWaiter registers its session inside register_wait(), while the
        # global filter list is intentionally owned here so replacement is safe.
        FILTERS.append(selection_filter)
        selection.task = asyncio.create_task(
            self._run_selection_wait(
                source_event=event,
                snapshot=snapshot,
                selection_filter=selection_filter,
                waiter=waiter,
                selection=selection,
            ),
            name=f"bili-player-selection-{snapshot.search_id}",
        )
        # Ensure SessionWaiter has registered before the result list can reach
        # a user able to reply immediately.
        await asyncio.sleep(0)
        return True

    async def _run_selection_wait(
        self,
        *,
        source_event: AstrMessageEvent,
        snapshot: SearchSnapshot,
        selection_filter: _SelectionSessionFilter,
        waiter: SessionWaiter,
        selection: _SelectionWait,
    ) -> None:
        async def select_song(
            controller: SessionController, reply: AstrMessageEvent
        ) -> None:
            await self._deliver_selection(controller, reply, snapshot)

        try:
            await waiter.register_wait(select_song, timeout=INTERACTION_TIMEOUT_SECONDS)
        except TimeoutError:
            if not selection.cancelled and self._initialized:
                await source_event.send(
                    source_event.plain_result("没有收到选歌回复，本次搜索已结束。")
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("bili-player selection wait failed")
            if self._initialized:
                await source_event.send(
                    source_event.plain_result("选歌流程已结束，请重新搜索。")
                )
        finally:
            try:
                FILTERS.remove(selection_filter)
            except ValueError:
                pass
            self._finish_selection_wait(snapshot.session_id, selection)

    async def _deliver_selection(
        self,
        controller: SessionController,
        reply: AstrMessageEvent,
        snapshot: SearchSnapshot,
    ) -> None:
        """Resolve the next real user reply against its original snapshot."""

        stop_wait = True
        settings = getattr(self, "_settings", None) or PluginSettings()
        try:
            if " ".join(reply.message_str.split()) == "取消":
                await reply.send(reply.plain_result("已取消选歌。"))
                return
            parsed = _parse_selection_for_settings(reply.message_str, settings)
            if parsed is None:
                await reply.send(reply.plain_result(_selection_help(settings)))
                stop_wait = False
                return
            position, action = parsed
            if action is _DeliveryMode.VIDEO and not settings.video_allowed:
                await reply.send(reply.plain_result("当前插件配置未开启视频交付。"))
                return
            if action is not _DeliveryMode.VIDEO and not settings.audio_allowed:
                await reply.send(reply.plain_result("当前插件配置未开启音频交付。"))
                return
            current = self._require_search().snapshot(
                search_id=snapshot.search_id,
                session_id=reply.unified_msg_origin,
            )
            candidate = current.candidate_at(position) if current is not None else None
            if candidate is None:
                await reply.send(reply.plain_result("搜索结果已过期，请重新搜索。"))
                return
            if _selection_requires_download(candidate, action, settings.limits.voice):
                voice_minutes = (settings.limits.voice.max_duration_ms or 0) // 60_000
                if settings.video_allowed:
                    hint = (
                        f"第 {position} 首音频超过 {voice_minutes} 分钟，无法直接播放；"
                        f"请回复“{position}”发视频，或回复“{position} 音频下载”下载音频文件。"
                    )
                else:
                    hint = (
                        f"第 {position} 首音频超过 {voice_minutes} 分钟，无法直接播放；"
                        f"请回复“{position} 音频下载”下载音频文件。"
                    )
                await reply.send(reply.plain_result(hint))
                stop_wait = False
                return
            await self._deliver_media(reply, candidate=candidate, action=action)
        except (DeliveryError, FfmpegUnavailableError, MediaError) as exc:
            await reply.send(reply.plain_result(str(exc)))
        except Exception:
            logger.exception("bili-player interactive delivery failed")
            await reply.send(reply.plain_result("歌曲发送失败，请稍后重试。"))
        finally:
            if stop_wait:
                controller.stop()

    async def _cancel_selection_wait(self, session_id: str) -> None:
        """Stop a just-created waiter when its result list could not be sent."""

        async with self._selection_lock:
            selection = self._selection_waits.get(session_id)
            if selection is not None:
                selection.cancelled = True
                selection.controller.stop()
        if selection is not None:
            await selection.finished.wait()

    async def account_status(self):
        owner = self._dashboard_owner()
        if owner is None:
            return error_response(
                "仅 Dashboard 管理员可管理账号与运行状态", status_code=403
            )
        accounts, media = self._require_accounts(), self._require_media()
        payload = await accounts.status_payload()
        payload["health"] = media.health.as_payload()
        settings = getattr(self, "_settings", None) or PluginSettings()
        payload["delivery"] = {
            "default_media": settings.default_media,
            "audio_form": settings.preferred_audio_form,
            "video_size_mb": settings.limits.video.max_bytes // (1024 * 1024),
            "download_size_mb": settings.limits.download.max_bytes // (1024 * 1024),
        }
        return json_response(payload)

    async def account_login(self):
        owner = self._dashboard_owner()
        if owner is None:
            return error_response(
                "仅 Dashboard 管理员可管理账号与运行状态", status_code=403
            )
        try:
            snapshot = await self._require_accounts().start_login(owner)
            try:
                return json_response(_public_login_snapshot(snapshot, include_qr=True))
            except Exception:
                await self._require_accounts().cancel_login(
                    snapshot["session_id"], owner
                )
                raise
        except AccountError as exc:
            return error_response(str(exc), status_code=400)
        except Exception:
            logger.exception("bili-player account login start failed")
            return error_response("无法创建登录二维码", status_code=502)

    async def account_events(self, session_id: str):
        owner = self._dashboard_owner()
        if owner is None:
            return error_response(
                "仅 Dashboard 管理员可管理账号与运行状态", status_code=403
            )
        events = self._require_accounts().login_events(session_id, owner)
        try:
            first = await anext(events)
        except LoginSessionNotFound as exc:
            return error_response(str(exc), status_code=404)
        except LoginSessionAccessDenied as exc:
            return error_response(str(exc), status_code=403)
        except AccountError as exc:
            return error_response(str(exc), status_code=400)

        async def stream() -> AsyncIterator[str]:
            yield _sse_data(_public_login_snapshot(first))
            async for snapshot in events:
                yield _sse_data(_public_login_snapshot(snapshot))

        return stream_response(
            stream(),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def account_cancel(self, session_id: str):
        owner = self._dashboard_owner()
        if owner is None:
            return error_response(
                "仅 Dashboard 管理员可管理账号与运行状态", status_code=403
            )
        try:
            await self._require_accounts().cancel_login(session_id, owner)
            return json_response({"session_id": session_id, "state": "cancelled"})
        except LoginSessionNotFound as exc:
            return error_response(str(exc), status_code=404)
        except LoginSessionAccessDenied as exc:
            return error_response(str(exc), status_code=403)
        except AccountError as exc:
            return error_response(str(exc), status_code=400)

    async def account_logout(self):
        owner = self._dashboard_owner()
        if owner is None:
            return error_response(
                "仅 Dashboard 管理员可管理账号与运行状态", status_code=403
            )
        try:
            await self._require_accounts().logout()
            return json_response({"state": "anonymous"})
        except AccountError as exc:
            return error_response(str(exc), status_code=400)

    async def account_import_credentials(self):
        owner = self._dashboard_owner()
        if owner is None:
            return error_response(
                "仅 Dashboard 管理员可管理账号与运行状态", status_code=403
            )
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            payload = {}
        cookie_text = str(payload.get("cookies") or "").strip()
        if not cookie_text:
            return error_response("请提供完整的 Cookie 字符串", status_code=400)
        try:
            profile = await self._require_accounts().import_credentials(cookie_text)
        except AccountError as exc:
            return error_response(str(exc), status_code=400)
        except Exception:
            logger.exception("bili-player account credential import failed")
            return error_response("无法导入账号凭证", status_code=502)
        return json_response(
            {"state": "connected", "display_name": profile.display_name}
        )

    async def _start_bilibili_login(self) -> QrLoginStart:
        qr_session = await self._require_bilibili().start_qr_login()
        return QrLoginStart(
            qr_url=qr_session.qr_url,
            poll_context=qr_session.poll_token,
            expires_at=qr_session.expires_at,
        )

    async def _poll_bilibili_login(self, poll_context: object) -> QrLoginPoll:
        result = await self._require_bilibili().poll_qr_login(str(poll_context))
        return QrLoginPoll(
            state=result.state,
            message=result.message,
            cookies=result.cookies,
        )

    async def _cancel_bilibili_login(self, poll_context: object) -> None:
        await self._require_bilibili().cancel_qr_login(str(poll_context))

    async def _bilibili_profile(self, cookies: dict[str, str]) -> Any:
        return await self._require_bilibili().profile_from_cookies(cookies)

    async def _send_onebot_voice(
        self, event: AstrMessageEvent, media: LocalMedia
    ) -> bool:
        """Send one local voice as a compact OneBot record segment on aiocqhttp.

        AstrBot's aiocqhttp adapter base64-encodes local Record components,
        and a full song can exceed NapCat's WebSocket payload limit. Passing
        the local file URI straight to the OneBot API keeps the message tiny
        when AstrBot shares its data directory with the OneBot implementation.
        """
        if event.get_platform_name() != "aiocqhttp":
            return False
        api = getattr(getattr(event, "bot", None), "api", None)
        if api is None or not callable(getattr(event, "is_private_chat", None)):
            return False
        payload: dict[str, Any] = {
            "message": [
                {
                    "type": "record",
                    "data": {"file": Path(media.path).resolve().as_uri()},
                }
            ]
        }
        if event.is_private_chat():
            payload["user_id"] = event.get_sender_id()
            await asyncio.wait_for(
                api.call_action("send_private_msg", **payload), timeout=45.0
            )
        else:
            payload["group_id"] = event.get_group_id()
            await asyncio.wait_for(
                api.call_action("send_group_msg", **payload), timeout=45.0
            )
        return True

    async def _deliver_candidate(
        self,
        candidate: BilibiliCandidate,
        action: _DeliveryMode,
    ) -> DeliveryResult:
        """Prepare the transport matching the selected action."""

        delivery = self._require_delivery()
        settings = getattr(self, "_settings", None) or PluginSettings()
        limits = _media_limits_for(action, settings.limits)
        if action is _DeliveryMode.VIDEO:
            return await delivery.deliver_video(candidate, limits=limits)
        return await delivery.deliver(
            candidate,
            limits=limits,
            prefer_highest=action is _DeliveryMode.DOWNLOAD,
        )

    async def _send_delivery(
        self,
        event: AstrMessageEvent,
        result: DeliveryResult,
        action: _DeliveryMode,
    ) -> None:
        media = self._require_media()
        try:
            platform_name = event.get_platform_name()
            if (
                action is _DeliveryMode.VOICE
                and platform_name not in _VOICE_AS_FILE_PLATFORMS
            ):
                try:
                    if await self._send_onebot_voice(event, result.media):
                        return
                    component = Record.fromFileSystem(result.media.path)
                    await event.send(MessageChain([component]))
                    return
                except Exception:
                    logger.warning(
                        "bili-player voice delivery failed on %s; falling back to file",
                        platform_name,
                    )

            if action is _DeliveryMode.VIDEO:
                try:
                    component = Video.fromFileSystem(result.media.path)
                    await event.send(MessageChain([component]))
                    return
                except Exception:
                    logger.warning(
                        "bili-player video delivery failed on %s; falling back to file",
                        platform_name,
                    )

            component = File(name=result.media.filename, file=str(result.media.path))
            await event.send(MessageChain([component]))
        finally:
            await media.release(result.media)

    def _register_account_routes(self) -> None:
        prefix = f"/{PLUGIN_NAME}/accounts"
        self.context.register_web_api(
            f"{prefix}/status",
            self.account_status,
            ["GET"],
            "Listen Music account status",
        )
        self.context.register_web_api(
            f"{prefix}/login",
            self.account_login,
            ["POST"],
            "Start music account QR login",
        )
        self.context.register_web_api(
            f"{prefix}/login/<session_id>/events",
            self.account_events,
            ["GET"],
            "Stream music account QR login status",
        )
        self.context.register_web_api(
            f"{prefix}/login/<session_id>/cancel",
            self.account_cancel,
            ["POST"],
            "Cancel music account QR login",
        )
        self.context.register_web_api(
            f"{prefix}/logout",
            self.account_logout,
            ["POST"],
            "Remove music account credentials",
        )
        self.context.register_web_api(
            f"{prefix}/credentials",
            self.account_import_credentials,
            ["POST"],
            "Import Bilibili cookies manually",
        )

    def _unregister_account_routes(self) -> None:
        """Remove handlers bound to this instance when AstrBot disables it."""

        routes = self.context.registered_web_apis
        routes[:] = [
            route
            for route in routes
            if not (
                route[0].startswith(f"/{PLUGIN_NAME}/accounts")
                and getattr(route[1], "__self__", None) is self
            )
        ]

    async def _replace_selection_wait(
        self, session_id: str, selection: _SelectionWait
    ) -> bool:
        """Replace a chat's host waiter only after the old one has cleaned up.

        AstrBot stores waiters globally by unified message origin.  Registering
        a replacement before the old waiter's final cleanup would let that
        cleanup remove the new waiter, so cancellation and replacement need a
        tiny serialized hand-off.
        """

        async with self._selection_lock:
            if not self._initialized:
                return False
            previous = self._selection_waits.get(session_id)
            if previous is not None:
                previous.cancelled = True
                previous.controller.stop()
                await previous.finished.wait()
            self._selection_waits[session_id] = selection
            return True

    def _finish_selection_wait(
        self, session_id: str, selection: _SelectionWait
    ) -> None:
        selection.finished.set()
        if self._selection_waits.get(session_id) is selection:
            self._selection_waits.pop(session_id, None)

    def _begin_llm_search(self, session_id: str) -> _LlmSearch:
        """Open one hidden search lease per chat without a background cleaner.

        A second ``find_in_bilibili`` call inside the same user turn means the
        LLM is fanning out one debounced multi-task prompt. Reject it instead
        of silently replacing the first search lease.
        """

        now = time.monotonic()
        self._purge_expired_llm_searches(now)
        active = self._llm_searches.get(session_id)
        if active is not None and active.expires_at > now:
            raise ValueError(
                "当前会话已有未完成的搜索；请先完成交付，或让用户重新发送一条消息。",
            )
        if (
            session_id not in self._llm_searches
            and len(self._llm_searches) >= SEARCH_SNAPSHOT_MAX_ENTRIES
        ):
            oldest_session = min(
                self._llm_searches,
                key=lambda key: self._llm_searches[key].expires_at,
            )
            self._llm_searches.pop(oldest_session, None)
        lease = _LlmSearch(expires_at=now + SEARCH_SNAPSHOT_TTL_SECONDS)
        self._llm_searches[session_id] = lease
        return lease

    def _complete_llm_search(
        self,
        session_id: str,
        lease: _LlmSearch,
        snapshot: SearchSnapshot,
        delivery_key: str,
    ) -> bool:
        """Publish results only when the request still owns this chat's lease."""

        if self._llm_searches.get(session_id) is not lease:
            return False
        lease.search_id = snapshot.search_id
        lease.expires_at = snapshot.expires_at
        lease.delivery = delivery_key
        return True

    def _is_current_llm_search(self, session_id: str, lease: _LlmSearch) -> bool:
        return self._llm_searches.get(session_id) is lease

    def _active_llm_search(self, session_id: str, search_id: str) -> _LlmSearch | None:
        """Return a live exact-match lease without consuming a newer request."""

        lease = self._llm_searches.get(session_id)
        if lease is None:
            return None
        if lease.expires_at <= time.monotonic():
            self._discard_llm_search(session_id, lease)
            return None
        if not search_id or lease.search_id != search_id:
            return None
        return lease

    def _consume_llm_search(self, session_id: str, lease: _LlmSearch) -> bool:
        if self._llm_searches.get(session_id) is not lease:
            return False
        self._llm_searches.pop(session_id, None)
        return True

    def _discard_llm_search(self, session_id: str, lease: _LlmSearch) -> None:
        if self._llm_searches.get(session_id) is lease:
            self._llm_searches.pop(session_id, None)

    def _purge_expired_llm_searches(self, now: float) -> None:
        expired = [
            session_id
            for session_id, lease in self._llm_searches.items()
            if lease.expires_at <= now
        ]
        for session_id in expired:
            self._llm_searches.pop(session_id, None)

    def _clear_llm_search(self, session_id: str) -> None:
        """Forget a model-visible candidate set when its conversation moves on."""

        searches = getattr(self, "_llm_searches", None)
        if searches is not None:
            searches.pop(session_id, None)

    def _dashboard_owner(self) -> str | None:
        config = self.context.get_config()
        dashboard = config.get("dashboard", {}) if hasattr(config, "get") else {}
        owner = dashboard.get("username") if isinstance(dashboard, dict) else None
        username = request.username
        if isinstance(owner, str) and owner.strip() and username == owner:
            return owner
        return None

    def _require_accounts(self) -> AccountService:
        if self._accounts is None:
            raise RuntimeError("plugin is not initialized")
        return self._accounts

    def _require_bilibili(self) -> BilibiliClient:
        if self._bilibili is None:
            raise RuntimeError("plugin is not initialized")
        return self._bilibili

    def _require_media(self) -> MediaStore:
        if self._media is None:
            raise RuntimeError("plugin is not initialized")
        return self._media

    def _require_search(self) -> SearchService:
        if self._search is None:
            raise RuntimeError("plugin is not initialized")
        return self._search

    def _require_delivery(self) -> DeliveryService:
        if self._delivery is None:
            raise RuntimeError("plugin is not initialized")
        return self._delivery

    def _format_selection_results(self, snapshot: SearchSnapshot) -> str:
        """Render the candidate catalogue for the currently configured actions."""

        settings = getattr(self, "_settings", None) or PluginSettings()
        return format_search_results(
            snapshot,
            video_enabled=settings.video_allowed,
            audio_enabled=settings.audio_allowed,
            default_audio_form=settings.preferred_audio_form,
            fuzzy_query=bool(getattr(snapshot, "fuzzy_query", False)),
            voice_max_duration_ms=settings.limits.voice.max_duration_ms,
            show_uploader=settings.show_uploader,
        )


def _parse_selection(message: str) -> tuple[int, _DeliveryMode] | None:
    """Parse the one user-facing grammar used by every selection flow.

    The public grammar is three commands: a bare number sends the video,
    "序号 音频" plays audio, and "序号 音频下载" downloads the audio file.
    The parser also tolerates natural synonyms such as "下载/听/播放" so an
    intent-bearing reply is never misread as a bare video request.
    """
    match = _SELECTION_RE.fullmatch(" ".join(message.split()))
    if match is None:
        return None
    prefix_action, raw_position, suffix_action = match.groups()
    actions = {
        _selection_mode_from_word(word)
        for word in (prefix_action, suffix_action)
        if word is not None
    }
    if len(actions) > 1:
        return None
    return _SELECTION_POSITION_MAP[raw_position], next(
        iter(actions), _DeliveryMode.VIDEO
    )


def _parse_selection_for_settings(
    message: str, settings: PluginSettings
) -> tuple[int, _DeliveryMode] | None:
    """Parse a reply with the configured default for a bare position number."""

    normalized = " ".join(message.split())
    match = _SELECTION_RE.fullmatch(normalized)
    if match is None:
        return None
    prefix_action, raw_position, suffix_action = match.groups()
    actions = {
        _selection_mode_from_word(word)
        for word in (prefix_action, suffix_action)
        if word is not None
    }
    if len(actions) > 1:
        return None
    action = next(iter(actions), _DeliveryMode.VIDEO)
    if action is _DeliveryMode.VIDEO and not actions and not settings.video_allowed:
        action = _action_for_delivery_key("audio", settings)
    return _SELECTION_POSITION_MAP[raw_position], action


def _selection_help(settings: PluginSettings) -> str:
    """One-line reply grammar rendered from the configured action surface."""

    options: list[str] = []
    if settings.video_allowed:
        options.append("“序号”发视频")
    if settings.audio_allowed:
        if settings.preferred_audio_form == "file":
            options.append(
                "“序号 音频”发音频文件"
                if settings.video_allowed
                else "“序号”或“序号 音频”发音频文件"
            )
        else:
            options.append(
                "“序号 音频”播放音频"
                if settings.video_allowed
                else "“序号”或“序号 音频”播放音频"
            )
            options.append("“序号 音频下载”下载音频文件")
    return "请回复：" + "；".join(options) + "。"


def _is_selection_reply(message: str) -> bool:
    return message == "取消" or _parse_selection(message) is not None


def _resolve_delivery_key(text: object, settings: PluginSettings) -> str:
    """Resolve the LLM's media intent against the configured policy."""

    key = " ".join(str(text or "").split()).casefold() or "auto"
    if key not in _DELIVERY_KEYS:
        key = "auto"
    if key == "auto":
        return settings.default_media
    if key == "video" and not settings.video_allowed:
        raise ValueError(
            "当前插件配置未开启视频交付。\n可在插件配置中调整“默认媒体类型”。"
        )
    if key in {"audio", "download"} and not settings.audio_allowed:
        raise ValueError(
            "当前插件配置未开启音频交付。\n可在插件配置中调整“默认媒体类型”。"
        )
    return key


def _action_for_delivery_key(
    delivery_key: str, settings: PluginSettings
) -> _DeliveryMode:
    """Turn the resolved intent into the first transport to attempt."""

    if delivery_key == "video":
        return _DeliveryMode.VIDEO
    if delivery_key == "download":
        return _DeliveryMode.DOWNLOAD
    if settings.preferred_audio_form == "file":
        return _DeliveryMode.DOWNLOAD
    return _DeliveryMode.VOICE


def _requires_user_page_choice(snapshot: SearchSnapshot) -> bool:
    """Require the user when the snapshot is an exact multi-page video or a fuzzy query."""

    return bool(
        getattr(snapshot, "fuzzy_query", False)
        or (
            getattr(snapshot, "by_video_reference", False)
            and len(snapshot.candidates) > 1
        )
    )


def _command_error_label(usage: str) -> str:
    """Keep command fallback errors consistent with the command the user sent."""

    if usage.startswith("搜索视频") or usage.startswith("/我要看"):
        return "搜索视频时发生错误"
    return "搜索歌曲时发生错误"


def _selection_mode_from_word(word: str) -> _DeliveryMode:
    if word in ("下载", "音频下载"):
        return _DeliveryMode.DOWNLOAD
    if word == "视频":
        return _DeliveryMode.VIDEO
    return _DeliveryMode.VOICE


def _selection_requires_download(
    candidate: BilibiliCandidate,
    action: _DeliveryMode,
    voice_limit: MediaLimits,
) -> bool:
    """Keep a long manual candidate available for its immediate file retry."""

    return bool(
        action is _DeliveryMode.VOICE
        and voice_limit.max_duration_ms is not None
        and candidate.duration_ms > voice_limit.max_duration_ms
    )


def _delivery_preparation_message(action: _DeliveryMode) -> str:
    """Return the sole user-visible status message before media preparation."""

    if action is _DeliveryMode.VIDEO:
        return "正在准备视频，请稍候。"
    if action is _DeliveryMode.DOWNLOAD:
        return "正在准备音频文件，请稍候。"
    return "正在准备音频，请稍候。"


def _public_login_snapshot(
    snapshot: dict[str, Any], *, include_qr: bool = False
) -> dict[str, Any]:
    public = dict(snapshot)
    qr_url = str(public.pop("qr_url", ""))
    if include_qr:
        public["qr_data_url"] = _qr_png_data_url(qr_url)
    return public


def _qr_png_data_url(content: str) -> str:
    if not content:
        raise ValueError("平台未返回有效登录二维码")
    try:
        import qrcode
    except ModuleNotFoundError as exc:
        raise RuntimeError("qrcode 依赖不可用") from exc
    image = qrcode.make(content)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
