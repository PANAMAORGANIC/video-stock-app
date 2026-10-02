# Clearview — architecture lock

Read this before changing anything. If a request fights this file, follow this file.

## What Clearview is

Personal footage NLE. Three layers, not three apps:

1. **Library** — media on disk (`storage/clips`, `storage/uploads`). Search, import, save. This is the only media store.
2. **Sequence** — the documentary document: ordered clips on a timeline (in/out, grade, speed, crop, text, markers). Director mutates this. **Assemble** is its view and its only whole-document writer. **Edit** works on ONE clip at a time and writes back either a new Library clip (`/api/clips/publish`) or a single scene (`/api/sequence/scene`).
3. **Export** — FFmpeg renders the Sequence. Never a second source of truth.

Product loop: Library → Edit (one clip) → Assemble (the documentary).

Canonical folder (do not copy the project):
`C:\Users\Downing\Downloads\video-stock-app\video-stock-app`

Ignore sibling copies (`video-stock-app (1)`, `video-stock-app-20th-century`). Do not create a third.

## Current debt (do not paper over)

Sequence is `storage/autosave/sequence.json` (version 1, `.bak` kept). Assemble loads/saves the Sequence. Edit writes one scene via `/api/sequence/scene` (or bakes a Library clip via `/api/clips/publish`). Disk wins; `pvs-clip-edition-v1` / `pvs-video-creation-v1` are cache only and migrate on read.

`POST /api/send-to-create` writes that sequence and opens Assemble. It does not bake new MP4s.

Every sequence write goes through `_sanitize_plan` (`remap_missing=False`). Do not split the project again.

## Hard rules

- Do not add features, pages, platforms, or Director verbs until Sequence is shared and tests pass.
- Do not add new `localStorage` keys. Do not add a fourth HTML app.
- Every plan/sequence write goes through `_sanitize_plan` (or a successor sanitizer). UI and Director never write raw scenes to FFmpeg.
- Director commands that cannot be sanitized return a reply and leave the sequence unchanged. Never IndexError, never empty-library crash.
- Media files stay on disk. Blob `blob:` URLs are not persistable; upload first (`/api/upload-media`) or drop the clip from save.
- Grade schema is `{lift, gamma, gain, sat, temp, contrast}`. Looks: Neutral, Film, Night, Golden, Punch. Keep Edit and Assemble on the same object.
- Spanish UI stays Spanish. Director understands Spanish and Premiere/Resolve language. Do not fork a second English app.
- Do not commit `.env`, cookies, Instagram sessions, or API keys.

## Tests (must stay green)

From `backend/`:

```
python -m unittest test_clearview -v
```

If you change Director, grade, plan sanitize, or image search, add a test first. A change that is not covered does not ship.

## Allowed work (in order)

1. Unify Sequence — done. Assemble is the whole-document writer; Edit is single-clip (`/api/clips/publish` or `/api/sequence/scene`).
2. Make Director + sanitize the only mutation path; expand `test_clearview.py`.
3. Point export at that Sequence.
4. Stage B (next, own commit): editor UI collapse — Edit stops sending the whole Sequence. Do not start it until this lock and Stage A tests are green.

## Brains and hands

Brain: the Clearview chat (architecture, go/no-go) and Director. They mutate Sequence only through `_sanitize_plan` / `save_sequence`. `storage/autosave/sequence.json` is the only document.

Hands: Grok Build CLI (code), FFmpeg (export), Edge TTS (`vo.mp3`), the player (preview). Hands execute the Sequence. They do not invent a second plan.

One writer: Assemble is the only writer of the whole document. Every Sequence write still goes through `save_sequence` and carries `rev` — including `/api/sequence/scene`. A stale Assemble tab (`rev` behind disk) is rejected (`source=stale`); disk wins. Edit must not POST a `scenes[]` document. Do not write `sequence.json` by hand. Do not edit the live tree while Grok Build is running. Default: architecture in this chat, code in Grok Build.

## Forbidden work

New stock platforms, new caption styles, new Imagine/TTS features, extra copies of the repo, rewriting the frontend in a new framework, "quick" parallel timelines.
