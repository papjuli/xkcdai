# xkcdai

An **MCP server** that surfaces the right [xkcd](https://xkcd.com) comic during a
conversation, if one is relevant.

> **Live connector:** `https://xkcdai.onrender.com/mcp` — add it in claude.ai →
> Settings → Connectors. See [Use the deployed MCP server](#use-the-deployed-mcp-server-as-custom-connector).

It builds a semantic index over every xkcd comic (title + mouseover text +
transcript) using on-device embeddings, then exposes a single `find_xkcd` tool.
A Claude conversation can call it whenever the topic feels xkcd-shaped; a
relevance threshold means weak matches return nothing, so it stays quiet instead
of forcing a tenuous reference.

The fetched transcripts, explanations, and the embeddings are currently committed in this repo, under [data/](data/).


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


## Use the deployed MCP server (as custom connector)

The server is deployed at **https://xkcdai.onrender.com** on Render. Add it as a Claude
**custom connector** to use it in the Claude web and mobile apps (note: the Free plan only allows one custom connector).
Anyone can add the same URL in their own account.

In **claude.ai** (web — do this once; it then syncs to the mobile app):

1. **Settings → Connectors → Add custom connector**.
2. Paste the connector URL, **including the `/mcp` path**: `https://xkcdai.onrender.com/mcp`
3. Leave OAuth blank (this server needs no auth) and click **Add**.
4. The connector's `find_xkcd` tool is now available in chats, on desktop and phone.
   For Claude to suggest comics on its own, also add the instruction from
   [Make Claude suggest comics proactively](#make-claude-suggest-comics-proactively)
   to your Profile preferences.

**Notes**
- The server is **public and unauthenticated** — fine here (read-only comic search,
  no secrets). Don't reuse this pattern for anything sensitive without OAuth.
- A free Render instance sleeps when idle, so the first request after a nap is
  slow (cold start + model load), then snappy.
- Hosted from this repo via the [Dockerfile](Dockerfile) and [render.yaml](render.yaml);
  pushes to `main` auto-redeploy.


## Local setup

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS/Linux:           source .venv/bin/activate
pip install -e .

# Fetch comics + their explainxkcd context, then embed (downloads the model once).
# First run ~10 min; re-running later only fetches what's new.
xkcdai build
```

Add `--no-enrich` to skip the explainxkcd fetch (faster/offline, weaker matches).

Test it from the command line:

```bash
xkcdai search "my code finally compiled after an hour"
xkcdai search "arguing about the correct date format"
xkcdai search "spent more time automating it than doing it by hand"
```


## Use locally as an MCP server

The server runs over stdio. Point your MCP host at it.

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "xkcdai": {
      "command": "C:\\your\\path\\to\\xkcdai\\.venv\\Scripts\\xkcdai-server.exe",
      "env": { "XKCDAI_DATA_DIR": "C:\\your\\path\\to\\xkcdai\\data" }
    }
  }
}
```

**Claude Code** (`-s user` makes it available in every project, not just this folder):

```bash
claude mcp add xkcdai -s user -e XKCDAI_DATA_DIR=C:\your\path\to\xkcdai\data -- C:\your\path\to\xkcdai\.venv\Scripts\xkcdai-server.exe
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


## Not just Claude — works with any MCP client

The examples above use Claude, but `find_xkcd` is a standard
[Model Context Protocol](https://modelcontextprotocol.io) tool, so any MCP-capable
host can use it. Point the client at either transport:

- **stdio:** run `xkcdai-server` locally (see [Use locally as an MCP server](#use-locally-as-an-mcp-server)), or
- **HTTP:** the deployed URL `https://xkcdai.onrender.com/mcp`.

The server is **LLM-agnostic** internally, too: matching runs on a local embedding model.
Only the host-specific bits differ — how you register the server, and where you put
the "suggest a comic when it fits" instruction (each client has its own
system-prompt / rules mechanism, e.g. Cursor Rules or a VS Code `.instructions` file).


## Configuration

- `XKCDAI_DATA_DIR` — where `comics.json`, `explain.json`, `embeddings.npy`, and
  `index.json` live.
- `find_xkcd(context, max_results=3, min_score=0.62)` — lower `min_score` for more
  (looser) suggestions, raise it to be stricter.


## Maintenance

Re-run `xkcdai build` periodically to pick up new comics — it incrementally fetches
new comics and their explainxkcd context, then re-embeds:

```bash
xkcdai build
```

Use `--force` to rebuild everything from scratch, or `--no-enrich` to skip the
explainxkcd fetch. `xkcdai enrich` fetches only the explainxkcd context.


## Credits & licensing

This project bundles content from two sources, each under its own license, so the
**code** and the **data** are licensed separately:

- **Code** (`src/`, `Dockerfile`, etc.) — [MIT](LICENSE).
- **Comics & mouseover text** — © [Randall Munroe / xkcd](https://xkcd.com),
  licensed [CC BY-NC 2.5](https://xkcd.com/license.html): **non-commercial**, with
  attribution.
- **Transcripts & explanations** (cached in [data/](data/)) — from
  [explainxkcd.com](https://www.explainxkcd.com), licensed
  [CC BY-SA 3.0](https://www.explainxkcd.com/wiki/index.php/explain_xkcd:Copyrights):
  redistributed here under the same license, with attribution.

Because `data/` mixes xkcd's NonCommercial content with explainxkcd's ShareAlike
content, treat the **data as non-commercial** and keep any redistribution under
these terms. The MIT license covers the source code only — not `data/`. At
runtime, `find_xkcd` results link back to both xkcd and explainxkcd for per-item
attribution.

This is an unofficial fan project, not affiliated with or endorsed by xkcd or explainxkcd.
