# ODYSSEUS API MANUAL — COMMAND SURFACE



Generated 2026-06-06T08:28:59Z


## 2.0 API ROUTES — COMMAND SURFACE

| # | Route Module | Purpose |
|---|--------------|---------|
| 1 | `auth_routes.py` | Authentication — login, logout, 2FA, API tokens |
| 2 | `chat_routes.py` | Chat — streaming LLM responses, session management |
| 3 | `email_routes.py` | Email — IMAP read/send, drafts, folders |
| 4 | `calendar_routes.py` | Calendar — CalDAV events, reminders, recurrence |
| 5 | `contacts_routes.py` | Contacts — CardDAV address book |
| 6 | `note_routes.py` | Notes — CRUD, tags, search |
| 7 | `memory_routes.py` | Memory — persistent user facts, categories |
| 8 | `document_routes.py` | Documents — upload, convert, search, PDF forms |
| 9 | `search_routes.py` | Search — web search via SearXNG/DuckDuckGo |
| 10 | `research_routes.py` | Research — deep multi-step research agent |
| 11 | `cookbook_routes.py` | Cookbook — model serve/download/manage |
| 12 | `gallery_routes.py` | Gallery — image viewing/management |
| 13 | `preset_routes.py` | Presets — saved chat model presets |
| 14 | `task_routes.py` | Tasks — todo/checklist management |
| 15 | `session_routes.py` | Sessions — chat session CRUD |
| 16 | `model_routes.py` | Models — model discovery/configuration |
| 17 | `mcp_routes.py` | MCP — Model Context Protocol server management |
| 18 | `skills_routes.py` | Skills — skill importer/manager |
| 19 | `shell_routes.py` | Shell — terminal command execution |
| 20 | `backup_routes.py` | Backup — data export/import |
| 21 | `webhook_routes.py` | Webhook — generic webhook receiver |
| 22 | `tts_routes.py` | TTS — text-to-speech |
| 23 | `stt_routes.py` | STT — speech-to-text |
| 24 | `upload_routes.py` | Upload — file upload handling |
| 25 | `codex_routes.py` | Codex — scope-gated agent API (Claude/Codex) |

> **Receipt:** 49 route files in `01_REPOS/odysseus/routes/`



## Full Routes

See `01_REPOS/odysseus/routes/` for all 49 route modules.
