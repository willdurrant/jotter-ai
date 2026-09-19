# jotter-ai

A small, provider-neutral RAG toolkit for Go: chunking, four retrievers, query
rewriting, and a grounded answer engine.

It came out of a tuning bench, so most of what is interesting is in the
comments — they record what was measured, on a real corpus, and why a default
is the number it is rather than a rounder one.

```
go get github.com/willdurrant/jotter-ai
```

## Layout

| Package | What it holds |
|---|---|
| `llm` | `Embedder`, `Chatter`, `Message`, `Usage` — the provider seam |
| `rag` | The measured defaults, and why each is what it is |
| `rag/corpus` | Loading documents from disk, and per-format normalisation |
| `rag/chunk` | What a chunk is, and the `Chunker` interface |
| `rag/store` | **The storage seam** — four capability interfaces |
| `rag/retrieve` | Vector, keyword, BM25 and hybrid retrieval; fusion; scoring |
| `rag/rewrite` | Query rewriting, including a standalone-query prompt and HyDE |
| `rag/answer` | The answer engine, and the instruction pair it sends |
| `ext/…` | Every heavy binding: langchaingo, pgvector, MuPDF |

**`llm/` and `rag/` depend on nothing outside the standard library.** That is
enforceable and worth enforcing:

```
go list -deps ./llm/... ./rag/... | grep -E '^(github|golang|gitlab)\.'
```

should return only this module's own packages. Everything that needs a driver,
a model client or a C toolchain lives under `ext/`, so importing the retrieval
logic does not drag in a database or cgo. In particular `ext/fitzpdf` needs
MuPDF and will not link under `CGO_ENABLED=0`; nothing else in the module
imports it.

## The storage seam

`rag/store` is four interfaces rather than one, because backends differ in what
they can honestly do:

| Interface | Needs | Gives you |
|---|---|---|
| `Searcher` | a vector index | vector retrieval |
| `Lexical` | a full-text index | keyword retrieval (`ts_rank`) |
| `Fusable` | both, joinable by row id | hybrid retrieval |
| `Lexicon` | the backend's own tokeniser | BM25 |

A table with vectors and no full-text index implements `Searcher` and nothing
else — and a caller asking it for BM25 fails to compile rather than failing in
front of a user. `ext/lcpgstore` implements all four over langchaingo's
pgvector schema.

## Scoring is pure

`FuseAlpha`, `Normalise` and `BM25Score` are plain functions over plain data,
testable with no database. Only tokenisation, matching and row retrieval sit
below the seam — BM25 is split across it deliberately, because gathering the
term statistics is SQL and scoring them is arithmetic.

## Status

v0.x. The API will move. `ext/lcpgstore` has no tests of its own here: it is
exercised against a real corpus in a private benchmark, which is where the
documents and question sets live.

## Licence

MIT.
