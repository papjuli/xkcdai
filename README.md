# xkcdai

An **MCP server** that surfaces the right [xkcd](https://xkcd.com) comic during a
conversation, if one is relevant.

It builds a local semantic index over every xkcd comic (title + mouseover text +
transcript) using on-device embeddings, then exposes a single `find_xkcd` tool.
A Claude conversation can call it whenever the topic feels xkcd-shaped; a
relevance threshold means weak matches return nothing, so it stays quiet instead
of forcing a tenuous reference.

## How it works

```
xkcd JSON API ─┐
               ├─► comics.json + explain.json ─► embeddings.npy ─► find_xkcd ─► Claude
explainxkcd  ──┘        (cache)                    (bge-small)     (cosine)    (mentions it
 (transcripts +                                                                 if it fits)
  explanations)
```

- **Data:** title + mouseover alt from the official API, plus the community
  **transcript** and **explanation** from [explainxkcd.com](https://www.explainxkcd.com).
  The explainxkcd context is essential: the official API dropped transcripts
  around comic ~1675, so without it the most-shared modern comics (e.g. #2347
  *Dependency*) are unmatchable — their joke text lives only inside the image.
- **Embeddings:** `fastembed` (ONNX) with `BAAI/bge-small-en-v1.5` — local, free,
  offline after first download, no PyTorch. Swap the model in `src/xkcdai/embed.py`
  (e.g. `BAAI/bge-base-en-v1.5` for marginally better ranking at ~3× the size).
- **Search:** a normalized numpy matrix + dot product. No vector DB needed for a
  few thousand comics.
- **Restraint:** because xkcd has a comic for *almost everything*, a similarity
  cutoff alone can't judge relevance. `min_score` (default `0.62`) is just a coarse
  floor; the real "should I bring this up?" decision is made by the calling model,
  guided by the score bands documented on the `find_xkcd` tool.

## Setup

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS/Linux:           source .venv/bin/activate
pip install -e .

# 1. Fetch every comic's metadata.
# 2. Fetch transcripts + explanations from explainxkcd (~2 min, be patient & polite).
# 3. Build the embedding index (downloads the model once; ~5-8 min to embed).
# Re-running later only fetches what's new.
xkcdai build          # fetch comic metadata
xkcdai enrich         # fetch explainxkcd context
xkcdai build          # embed everything into the index
```

Test it from the command line:

```bash
xkcdai search "my code finally compiled after an hour"
xkcdai search "arguing about the correct date format"
xkcdai search "spent more time automating it than doing it by hand"
```

## Use as an MCP server

The server runs over stdio. Point your MCP host at it.

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "xkcdai": {
      "command": "C:\\Users\\papju\\claude\\xkcdai\\.venv\\Scripts\\xkcdai-server.exe",
      "env": { "XKCDAI_DATA_DIR": "C:\\Users\\papju\\claude\\xkcdai\\data" }
    }
  }
}
```

**Claude Code** (`-s user` makes it available in every project, not just this folder):

```bash
claude mcp add xkcdai -s user -e XKCDAI_DATA_DIR=C:\Users\papju\claude\xkcdai\data -- C:\Users\papju\claude\xkcdai\.venv\Scripts\xkcdai-server.exe
```

Always set `XKCDAI_DATA_DIR`, since the host launches the server from an arbitrary
working directory.

> MCP only gives Claude the *ability* to call `find_xkcd` — it won't volunteer
> comics on its own. See [Make Claude suggest comics proactively](#make-claude-suggest-comics-proactively).

## Make Claude suggest comics proactively

Connecting the server only gives Claude the *ability* to call `find_xkcd`; it
won't reach for it unprompted. To make Claude volunteer comics, paste the
instruction below wherever that Claude reads persistent instructions:

- **Claude Code** — your global `~/.claude/CLAUDE.md` (applies everywhere) or a
  per-repo `CLAUDE.md`; restart the session to load changes.
- **Claude.ai / Claude Desktop** — Settings → Profile → *"What personal preferences
  should Claude consider in responses?"* (every plan, including free; syncs to the
  mobile app). Each person who uses the connector adds it in their own account.

```text
When a conversation naturally lands on a topic xkcd is known for — programming,
science, math, statistics, engineering, the absurdity of standards, relationships,
everyday life — call the find_xkcd tool (xkcdai) with a short phrase describing the
topic. Then judge whether to bring it up:
- score >= 0.75 — strong match; mention it if it fits the moment
- 0.66-0.75 — only if it genuinely lands
- below that — stay silent
When you share one, give just that single comic: its number and title, its URL, and
quote the alt (mouseover) text — that's half the joke. At most one comic per topic,
and never force a tangential reference. When in doubt, say nothing.
```

It's still Claude's judgment, so it won't fire on every borderline topic — asking
*"is there an xkcd for this?"* always triggers a lookup.

## Share it on a phone (host as a Claude custom connector)

A local stdio server only works on the machine it runs on. To use it on a phone,
host the **HTTP** build publicly and add it as a Claude **custom connector** — which
works in the Claude web and **mobile** apps (Free plan allows one connector; Pro/Max
more). You host it **once** and anyone can add the same URL in their own account.

The server already speaks HTTP when `XKCDAI_TRANSPORT=streamable-http`, serving MCP
at `/mcp` on `0.0.0.0:$PORT`. The [Dockerfile](Dockerfile) bakes in the prebuilt
index and embedding model. First make sure the index exists locally
(`xkcdai build && xkcdai enrich && xkcdai build`), then deploy:

**Option A — Fly.io** (builds remotely, so no local Docker, and ships the gitignored
`data/` straight from your folder):

```bash
fly launch --no-deploy      # creates fly.toml; set internal_port = 8000
fly deploy                  # -> https://<app>.fly.dev
```

**Option B — Render** (free, no card; needs the repo on GitHub *with the index*):

```bash
git init && git add -A && git add -f data/*.json && git commit -m "deploy"
# push to GitHub, then on render.com: New > Web Service > your repo,
# Runtime = Docker. Render gives https://<app>.onrender.com
```

Then, in **claude.ai** (web — do this once; it then syncs to the mobile app):

1. **Settings → Connectors → Add custom connector**.
2. Paste the server URL **with the `/mcp` path**, e.g. `https://<your-host>/mcp`.
3. Leave OAuth blank (this server needs no auth) and click **Add**.
4. On the phone, open the Claude app → in a chat the connector's `find_xkcd` tool
   is now available. (For it to fire proactively, add the instruction from
   [Make Claude suggest comics proactively](#make-claude-suggest-comics-proactively)
   to your claude.ai Profile preferences.)

Share the `https://<your-host>/mcp` URL with anyone — they repeat steps 1–4 in
their own Claude account.

**Notes**
- The server is **public and unauthenticated** — fine here (read-only comic search,
  no secrets). Don't put anything sensitive behind this pattern without OAuth.
- Free hosts sleep when idle, so the first request after a nap is slow (cold start
  + model load); it's snappy afterward.

## Configuration

- `XKCDAI_DATA_DIR` — where `comics.json`, `explain.json`, `embeddings.npy`, and
  `index.json` live.
- `find_xkcd(context, max_results=3, min_score=0.62)` — lower `min_score` for more
  (looser) suggestions, raise it to be stricter.

## Maintenance

Pick up new comics periodically (both steps are incremental):

```bash
xkcdai enrich     # new explainxkcd context
xkcdai build      # fetch new comics + re-embed
```

Use `xkcdai build --enrich` to do both in one go, or `--force` on either command
to rebuild everything from scratch.
