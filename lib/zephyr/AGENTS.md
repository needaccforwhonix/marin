# Zephyr Agent Notes

Lazy dataset processing library. Start with the shared instructions in `/AGENTS.md`; only Zephyr-specific conventions are below.

## Key Docs

- `README.md` — overview, API reference, quick start
- `OPS.md` — debugging pipelines: dashboard, observability, profiling, diagnostic patterns (also used by skills: `debug-infra`, `babysit-zephyr`)
- Archived: `.agents/project/20260130_fray_lite_design.md` — Fray v2 backend design (implemented; read `lib/fray/src/fray/v2/` instead)

## Source Layout

- `src/zephyr/dataset.py` — `Dataset` class, `ShardInfo`, all transformation operations including `group_by`, `deduplicate`, `sorted_merge_join`
- `src/zephyr/execution.py` — `ZephyrContext`, `ZephyrCoordinator`, `ZephyrWorker`, `CounterSnapshot`, execution control flow
- `src/zephyr/plan.py` — `compute_plan`, `PhysicalPlan`, operation fusion
- `src/zephyr/readers.py` — `load_jsonl`, `load_parquet`, `load_vortex`, `InputFileSpec`
- `src/zephyr/writers.py` — `write_jsonl_file`, `write_parquet_file`, `write_vortex_file`, Levanter cache writer
- `src/zephyr/shuffle.py` — scatter pipeline internals (`ScatterFileIterator`, `ScatterReader`, hash-routing, combiner, zstd-chunk file format with byte-range sidecar)
- `src/zephyr/expr.py` — `Expr`, `col`, `lit` for filter expressions
- `src/zephyr/external_sort.py` — `external_sort_merge` k-way merge of sorted runs
- `src/zephyr/counters.py` — `increment` / `get_counters` per-worker counter API (`CounterSnapshot` lives in `execution.py`)

## Execution Model

Actor-based, pull-based task distribution. Workers are persistent across stages.

```
ZephyrContext → ZephyrCoordinator (fray actor) → ZephyrWorker actors (fray actor_group)
```

### Data flow between stages

Stages pass data via **filesystem-backed chunk references** (`PickleDiskChunk`), not in-memory. Each stage reads chunks from storage, processes them, writes results back. Workers stream one chunk at a time to minimize memory.

### Critical `.result()` calls

These worker→coordinator RPCs **must** block (`.result()`). Removing them causes race conditions:

1. `coordinator.report_result.remote().result()` — must complete before next `pull_task`, otherwise `_in_flight` tracking breaks (assertion at line ~584)
2. `coordinator.report_error.remote().result()` — same ordering constraint as `report_result`
3. `coordinator.heartbeat.remote().result()` — prevents congesting the coordinator RPC pipe with fire-and-forget heartbeats
4. `coordinator.register_worker.remote().result()` — worker must be registered before polling starts

Shared data is uploaded to filesystem by `ZephyrContext._upload_shared_data()` before pipeline execution; workers read it lazily via `get_shared(name)`. Chunk config is passed inline with each `pull_task` response (not via a separate RPC).

### Error classification

- **Transient** (connection errors, preemption) → task re-queued
- **Permanent** (user code bugs, invalid data) → `fatal_error` set, exception raised

## Notes

### MacOS

Ray 2.53 enables a `uv run` runtime_env hook by default. When tests run via `uv run pytest`, this can start workers with a different Python version or fail with psutil errors in sandboxed environments. Disable it for tests. See https://github.com/ray-project/ray/issues/59639.



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
