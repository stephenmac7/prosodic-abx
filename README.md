# Prosodic ABX — Supplementary Material

Interactive supplementary material for:

> *Prosodic ABX: A Language-Agnostic Method for Measuring Prosodic Contrast in Speech Representations*
> Interspeech 2026

Hosted at: <https://stephenmac7.github.io/prosodic-abx/>

This is the `gh-pages` branch of the [`prosodic-abx`](https://github.com/stephenmac7/prosodic-abx) repository. It has no shared history with `master` (it is an orphan branch) and contains only the static site published via GitHub Pages.

## Contents

| Path | Description |
|---|---|
| `index.html` | Single-page interactive visualisation (Apache ECharts) |
| `data/data.js` | All result data bundled as `window.WEBPAGE_DATA` for `file://` compatibility |
| `data/models.json` | Model metadata (19 models: 17 S3Ms + MFCC/FBank baselines) |
| `data/results_natural.json` | Layer-wise ABX error rates on natural speech |
| `data/results_synth.json` | Layer-wise ABX error rates on synthesised speech (TTS proxy) |
| `data/results_incontext.json` | Out-of-context vs. in-context results (Japanese pitch accent) |

## Updating the data

From the root of the main research repo (`master` branch), run the export script:

```bash
uv run scripts/export_webpage_data.py
```

This reads result CSVs from `results/` and rewrites `data/data.js` and the four JSON files under the supplementary data directory.

## Publishing changes

The site is served from this `gh-pages` branch. To update it, commit the regenerated site files onto `gh-pages` and push:

```bash
git checkout gh-pages
# copy in the freshly exported index.html + data/ files
git add .
git commit -m "update supplementary site"
git push origin gh-pages
```

GitHub Pages redeploys automatically within a minute or two of each push.
