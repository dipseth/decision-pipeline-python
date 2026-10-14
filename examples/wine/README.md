# Wine: feature discovery into a spec (Python)

The Python port of the [wine example](https://github.com/dipseth/decision-pipeline/tree/main/examples/wine) from the TypeScript repo: TypeSafe's autoresearch feature-discovery cookbook re-run on this package, ending in a compiled spec whose decide port calls Jev. That README has the write-up and the numbers; the two scripts take the same flags and produce the same files.

```bash
uv sync --extra jsonata                       # once, in the repo root
python3 examples/wine/sample.py               # the cookbook's exact 2,000-review sample → data/wine.json
TYPESAFE_API_KEY=… ANTHROPIC_API_KEY=… uv run python examples/wine/run.py --run r1 [--rounds 5]
```

Every Jev answer is cached in `data/cache.jsonl` with the same key the TypeScript script uses, so a cache written by either language replays in the other.
