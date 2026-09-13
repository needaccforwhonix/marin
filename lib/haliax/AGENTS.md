# Haliax LLM Agent Guidelines

This document summarizes important conventions for contributing code or documentation to the Haliax
repository. Follow these notes when implementing new features or fixing bugs.

## General Guidelines

* **Get better.** Whenever you discover something missing from these guidelines, or the requester
  suggests a better way to do something, please update this document. The goal is to make it easier for
  everyone to contribute and maintain the codebase. Generally speaking, you should add bullets or new sections.
  Be sure to do this when directed to. For example, if directed that you should never relax tolerances in
  floating point tests, add that to the list.
* **Playbooks.** Sometimes, there are repeatable tasks (e.g. porting models) for which we follow a standard set of steps.
  Please reference `.playbooks/` to see what playbooks are available, or see the list below. If you want to add a playbook
  write a markdown doc named e.g. `.playbooks/add-types.md` and add a pointer to it in the list below.

## Playbook

- Adding Haliax-style tensor typing annotations are described in @.playbooks/add-types.md
- [Wrapping standard JAX functions](.playbooks/wrap-non-named.md) so they operate on `NamedArray`

## Code Style

* **Python version**: the project targets Python >=3.10.
* **Formatting and Linting**: We use `./infra/pre-commit.py` (ruff, black, license headers) to keep files consistent.
* **Typing**: the code base uses `mypy` for static type checking. `mypy` is run by the same `infra/pre-commit.py` entrypoint and the
  configuration is found in `pyproject.toml`.
* **Run `./infra/pre-commit.py --all-files`** before committing. The CI workflows run the same checks.
* **Use `uv run` for commands.** When running tools like `pytest` or other scripts, invoke them via `uv run` so the development dependencies are active.
* **Doc Strings**: All public functions, classes, and modules should have docstrings, unless
  their purpose is painfully obvious. Use
  [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for
  consistency.
* **Commenting**: Use comments to explain why something is done a certain way, especially if it is not
  immediately obvious. Avoid commenting on every line of code; focus on the intent and purpose of
  complex logic. Demarcating logical groups of code with comments is encouraged, unless it is better
  to refactor the code into smaller functions or classes.
* **Mkdocs**: We use [Mkdocs](https://www.mkdocs.org/) for documentation. The main documentation is in
  the `docs` directory. Use Markdown for writing docs, and follow the existing structure. When linking to
  symbols, prefer using mkdocs-style links (e.g. With a custom title: `[full.path.object2][]` or
  `[Object 1][full.path.object1]`)
* **Documentation**: When adding new features, ensure that the documentation is updated accordingly.
  This includes updating the Mkdocs files and any relevant docstrings. If you add a new module or
  significant functionality, consider adding a dedicated section in the documentation. When you
  wrap a new JAX function, add a reference to it in `docs/api.md` so users can discover it.

## Testing

* Tests are executed with `pytest`. The default workflow runs ` XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=tests:src:. uv run pytest tests`.
* In general, never relax tolerances in floating point tests unless specifically discussed with the
  team. Use `assert_allclose` with appropriate tolerances for numerical comparisons. We typically use
  1e-4 for more complex modules, and 1e-5 for simpler ones.
* Always mark tests that depend on pytorch with `@skip_if_no_torch` to ensure they are skipped
  when PyTorch is not available. This is particularly important for tests that require PyTorch-specific
  functionality.


## Design Preferences

* **Generic code**: many utilities are written with Python generics and dataclasses. Where possible,
  write reusable functions or classes that operate over TypeVars instead of hard coding concrete types.
* **Reproducibility**: Haliax aims for determinism where possible. Avoid sources of
  nondeterminism unless explicitly required.
* Prefer Stacked with fold or scan over writing custom loops, for better compile times and gradient checkpointing support
* For configuration, we prefer frozen dataclasses over dictionaries.

## Library conventions
- Haliax revolves around `NamedArray` and named shapes, either via Axis objects or "shape dicts" (e.g. `{"batch": 42, "embed": 16}).
  Prefer APIs that accept axes or axis names rather than hard‑coding positional dimensions. In particular, use AxisSpec and AxisSelection where possible.
- Utilities should be written so they work with arbitrary axis names. Avoid relying on
  fixed axis orders when possible.
- Use the provided modules in `haliax.nn` or Equinox when building neural network layers.
- Type annotations can use named shapes shorthand provided in `haliax.haxtyping`: `ht.f32[NamedArray, "batch"]`
  for a float32 array with a "batch" axis, or `ht.Float[NamedArray, "batch"]` for any floating point dtype.

## Documentation
- Public functions and modules require docstrings. If behavior is non‑obvious, add examples in `docs/`.
- For a concise overview of Haliax aimed at LLM agents, see [docs/primer.md](docs/primer.md).



## 🛡️ Strict Embedding Separation, Zero-Fallback Law & 24/7 Dual-GPU Invariant
- **Reference**: `/home/m1st/.agents/rules/RULE_Strict_Embedding_Separation_And_Dual_Pipeline.md`

### 1. Das Absolute Fallback-Verbot (Zero-Fallback Law)
Unter keinen Umständen, zu keinem Zeitpunkt und aus keinem Grund darf ein Fallback zwischen verschiedenen Embedding-Modellen stattfinden.
* **Geltende Aktion:** Fällt ein Embedding-Modell aus oder ist überlastet, MUSS die Operation sofort hart fehlschlagen (`Fail-Fast`) oder die Payload transaktional in einer Queue (NATS/SQLite) verharren, bis das exakte Modell bereit ist.
* **Verboten:** Kein stiller oder dynamischer Modellwechsel (weder Jina -> Gemma noch umgekehrt).

### 2. Warum ein Embedding-Fallback mathematisch & informationstheoretisch unmöglich ist
* **Topologische Inkompatibilität heterogener Vektorräume (Non-Isomorphism):**
  Jedes Modell $f_\theta: \mathcal{X} \to \mathbb{R}^D$ projiziert Text in eine spezifische, gelernte Riemannsche Mannigfaltigkeit. Jina v5 ($D=256$) und EmbeddingGemma ($D=768$) spannen zwei völlig inkompatible geometrische Räume auf. Die Basisvektoren der semantischen Achsen sind ohne explizite Procrustes-Transformation nicht ausgerichtet.
* **Kollaps der Kosinus-Ähnlichkeit ($	ext{sim} \approx 0$):**
  Wird eine Suchanfrage mit Modell $B$ berechnet ($v_q = f_B(q)$), während der Dokumentenkorpus mit Modell $A$ indiziert wurde ($v_d = f_A(d)$), verhält sich das Skalarprodukt mathematisch wie das zweier rein zufälliger Vektoren auf einer hochdimensionalen Einheitssphäre:
  $$\mathbb{E}[\text{sim}(u, v)] = 0 \quad \text{mit Varianz} \quad \sigma^2 = \frac{1}{D}$$
  Der Nearest-Neighbor-Algorithmus (HNSW/k-NN) liefert stochastisches Rauschen. Das RAG-System erhält völlig falsche oder irrelevante Kontexte.
* **Irreversible Index-Vergiftung (Index Poisoning):**
  Wird auch nur ein einziger Vektor von Modell $B$ als "Fallback" in den Index von Modell $A$ geschrieben, verunreinigt er die Distanzgraphen und Clusterzentren dauerhaft.
* **Das Gesetz des Fail-Fast:**
  Ein Ausfall muss hart abbrechen (`HTTP 503 Service Unavailable / IngestionQueueBlocked`).

### 3. Duale 24/7 Erfassungspflicht (GPU-Only)
* **GPU-Only Mandat:** Es läuft absolut nichts auf der CPU — GPU ONLY (NVIDIA GB10 CUDA) für ausnahmslos jedes Embedding-Modell.
* **24/7 Parallelität:** Sowohl `jina-embeddings-v5-omni-nano-classification` (256D, ~4,1 GB VRAM) als auch `google/embeddinggemma-300m` (768D, ~1,2 GB VRAM) laufen dauerhaft 24/7 im VRAM (Summe ~5,3 GB VRAM).
* **Duale Erfassung:** Jeder zu indizierende Text/Chunk wird immer von beiden Modellen parallel eingebettet und getrennt persistiert.

### 4. Idioten- & Failsafe-Sicherung auf Datenbankebene
* **SQLite Schema CHECK-Constraints:**
  `model_signature TEXT NOT NULL CHECK(model_signature = '...')` und `dimension INTEGER NOT NULL CHECK(dimension = ...)` erzwingen atomare Abbrüche auf Engine-Ebene bei Modell-Mismatches.
* **Qdrant Collection Constraints:**
  Strikte Trennung in separate Collections (`dgx_text_embeddings_jina_256` vs `dgx_text_embeddings_gemma_768`) mit fixierter Vektordimension.


## ⚡ High-Quality Systems Programming Languages Priority (No-Python Policy)
- **Reference**: `/home/m1st/.agents/rules/RULE_High_Quality_Systems_Programming_Languages.md`
- **Rule**:
  1. **Bevorzugte Sprachen:** High Quality **Golang (Go), Rust, C++, Zig, PowerShell, C** sind IMMER und AUSNAHMSLOS die bevorzugten Programmiersprachen.
  2. **Kein Python:** Python ist für neue Daemons, Watcher, Automatisierungen, APIs, CLI-Tools und Dienste strikt untersagt (GIL-Bottlenecks, Dependency-Drift, Speicherineffizienz).
  3. **Natives Systems-Engineering:** Alle Hintergrunddienste, Caching-Ebenen und Task-Runner müssen als native, speichersichere und nebenläufige Binaries kompiliert werden.
