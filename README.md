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

**Claude Code:**

```bash
claude mcp add xkcdai -- C:\Users\papju\claude\xkcdai\.venv\Scripts\xkcdai-server.exe
```

(or `python -m xkcdai.server` with the venv's Python). Set `XKCDAI_DATA_DIR` if
the index lives somewhere other than `./data`, since the host launches the server
from an arbitrary working directory.

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
