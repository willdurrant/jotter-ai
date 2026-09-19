// Package retrieve turns a query into ranked chunks. All scoring here is pure:
// the SQL lives below the store seam.
package retrieve

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/willdurrant/jotter-ai/rag"
	"github.com/willdurrant/jotter-ai/rag/store"
)

// Result is one retrieved chunk. Unlike langchaingo's schema.Document it
// carries a row identity, which is what makes fusing two retrievers' output
// possible without matching on document text.
type Result struct {
	ID     string
	Text   string
	Source string

	// Score is the retriever's own score. It is NOT comparable across
	// retriever types: cosine similarity for vector, ts_rank for keyword, and
	// a normalised blend for hybrid. Compare within a family only.
	Score float64

	VectorScore  float64
	KeywordScore float64

	// Meta is whatever the store attached to the row, passed through so a
	// caller's own vocabulary survives retrieval.
	Meta map[string]string
}

// Retriever finds the k chunks most relevant to a query.
type Retriever interface {
	Retrieve(ctx context.Context, query string, k int) ([]Result, error)
	Name() string
}

// --- vector ---------------------------------------------------------------

// VectorRetriever is the baseline: pure cosine similarity via langchaingo.
//
// The threshold is held here rather than passed to the backend as its own
// option type, so the Retriever interface stays free of backend-specific
// types. It travels as store.Query.MinScore.
type VectorRetriever struct {
	src       store.Searcher
	threshold float32
}

func NewVector(src store.Searcher, threshold float32) *VectorRetriever {
	return &VectorRetriever{src: src, threshold: threshold}
}

func (r *VectorRetriever) Name() string {
	if r.threshold > 0 {
		return fmt.Sprintf("vector(threshold=%.2f)", r.threshold)
	}
	return "vector"
}

func (r *VectorRetriever) Retrieve(ctx context.Context, query string, k int) ([]Result, error) {
	docs, err := r.src.Search(ctx, store.Query{
		Text:     query,
		K:        k,
		MinScore: float64(r.threshold),
	})
	if err != nil {
		return nil, err
	}

	out := make([]Result, 0, len(docs))
	for _, d := range docs {
		out = append(out, Result{
			ID:          d.ID,
			Text:        d.Text,
			Source:      d.Source,
			Meta:        d.Meta,
			Score:       d.Score,
			VectorScore: d.Score,
		})
	}
	return out, nil
}

// --- keyword --------------------------------------------------------------

// KeywordRetriever ranks by Postgres full-text search over the same document
// column the vectors were built from. No embedding call, so no API cost.
//
// Note this is ts_rank, not BM25: Postgres ranks by term frequency and cover
// density. True BM25 needs an extension such as ParadeDB's pg_search.
type KeywordRetriever struct {
	src store.Lexical
	// matchAll requires every query term to appear (AND). Off by default —
	// see the note on tsqueryAll above.
	matchAll bool
}

func NewKeyword(src store.Lexical) *KeywordRetriever {
	return &KeywordRetriever{src: src}
}

// NewKeywordAll requires every term to match, for comparison.
func NewKeywordAll(src store.Lexical) *KeywordRetriever {
	return &KeywordRetriever{src: src, matchAll: true}
}

func (r *KeywordRetriever) Name() string {
	if r.matchAll {
		return "keyword(AND)"
	}
	return "keyword(OR)"
}

func (r *KeywordRetriever) Retrieve(ctx context.Context, query string, k int) ([]Result, error) {
	rows, err := r.src.SearchKeyword(ctx, store.KeywordQuery{
		Text:     query,
		K:        k,
		MatchAll: r.matchAll,
	})
	if err != nil {
		return nil, err
	}

	var out []Result
	for _, d := range rows {
		out = append(out, Result{
			ID: d.ID, Text: d.Text, Source: d.Source, Meta: d.Meta,
			Score: d.Score, KeywordScore: d.Score,
		})
	}
	return out, nil
}

// --- hybrid ---------------------------------------------------------------

// HybridRetriever blends both signals. Alpha weights the vector side:
// 1.0 is pure vector, 0.0 is pure keyword.
type HybridRetriever struct {
	src      store.Fusable
	alpha    float64
	matchAll bool
}

func NewHybrid(src store.Fusable, alpha float64) *HybridRetriever {
	return &HybridRetriever{src: src, alpha: alpha}
}

func (r *HybridRetriever) Name() string { return fmt.Sprintf("hybrid(a=%.2f)", r.alpha) }

func (r *HybridRetriever) Retrieve(ctx context.Context, query string, k int) ([]Result, error) {
	// One query for both sides, joined on row identity by the backend. Two
	// separate searches could not be fused reliably before Row carried an ID,
	// and asking for them together still lets the backend use whatever join
	// its storage makes cheap.
	cands, err := r.src.SearchCandidates(ctx, store.CandidateQuery{
		Text:     query,
		Depth:    rag.DefaultCandidateDepth,
		MatchAll: r.matchAll,
	})
	if err != nil {
		return nil, err
	}

	out := make([]Result, 0, len(cands))
	for _, c := range cands {
		out = append(out, Result{
			ID: c.ID, Text: c.Text, Source: c.Source, Meta: c.Meta,
			VectorScore: c.VectorScore, KeywordScore: c.KeywordScore,
		})
	}

	return FuseAlpha(out, r.alpha, k), nil
}

// --- fusion ---------------------------------------------------------------

// FuseAlpha blends the two component scores and returns the top k.
//
// Each side is min-max normalised first: cosine similarity sits around
// 0.5-0.65 on this corpus while ts_rank sits near 0-0.3, so blending the raw
// values would let the vector side dominate through magnitude alone rather
// than through relevance.
func FuseAlpha(cands []Result, alpha float64, k int) []Result {
	vMin, vMax := extent(cands, func(r Result) float64 { return r.VectorScore })
	kMin, kMax := extent(cands, func(r Result) float64 { return r.KeywordScore })

	out := make([]Result, len(cands))
	copy(out, cands)
	for i := range out {
		v := Normalise(out[i].VectorScore, vMin, vMax)
		kw := Normalise(out[i].KeywordScore, kMin, kMax)
		out[i].Score = alpha*v + (1-alpha)*kw
	}

	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

func extent(rs []Result, get func(Result) float64) (min, max float64) {
	if len(rs) == 0 {
		return 0, 0
	}
	min, max = get(rs[0]), get(rs[0])
	for _, r := range rs[1:] {
		v := get(r)
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}

// Normalise maps v into 0..1. A degenerate range (every score equal, or a
// side that returned nothing at all) contributes 0 rather than NaN, so that
// side simply does not influence the ranking.
func Normalise(v, min, max float64) float64 {
	if max <= min {
		return 0
	}
	return (v - min) / (max - min)
}

// --- BM25 -----------------------------------------------------------------

// BM25Retriever ranks the same rows KeywordRetriever matches, by Okapi BM25
// instead of ts_rank.
//
// The point of it is IDF. ts_rank scores on term frequency and cover density
// and weights a term appearing in 890 of 893 chunks exactly like one appearing
// in a single chunk — which sits badly with this project's central lexical
// finding, that rare literal tokens are what keyword search is for. BM25 adds
// three things ts_rank lacks: inverse document frequency, term-frequency
// saturation (k1) and document-length normalisation (b).
//
// IT IS RANKED IN GO, DELIBERATELY. A true index-level BM25 needs an extension
// such as ParadeDB's pg_search, which would mean a different Postgres image AND
// a different tokeniser — and varying the tokeniser at the same time as the
// ranking function would make any difference unattributable. Here Postgres does
// the tokenising for both retrievers, via the same to_tsvector('english', ...),
// so the ONLY thing that differs is the arithmetic. On a corpus of this size
// that is also exact rather than approximate: every matching chunk is scored.
type BM25Retriever struct {
	src      store.Lexicon
	k1, b    float64
	matchAll bool

	stats *StatsCache
}

// NewBM25 uses the conventional parameters. Pass others to sweep them.
func NewBM25(src store.Lexicon) *BM25Retriever {
	return NewBM25With(src, rag.DefaultBM25K1, rag.DefaultBM25B)
}

func NewBM25With(src store.Lexicon, k1, b float64) *BM25Retriever {
	return &BM25Retriever{src: src, k1: k1, b: b, stats: NewStatsCache(src, 0)}
}

// WithStatsCache shares a cache between retrievers, or supplies one with a TTL.
// A sweep over k1 and b builds many retrievers over one unchanging index, and
// they have no reason to scan it once each.
func (r *BM25Retriever) WithStatsCache(c *StatsCache) *BM25Retriever {
	r.stats = c
	return r
}

func (r *BM25Retriever) Name() string {
	return fmt.Sprintf("bm25(k1=%.2f,b=%.2f)", r.k1, r.b)
}

// Retrieve scores the SAME rows KeywordRetriever would match — the identical
// @@ filter and tsquery — and differs from it only in how it orders them. That
// is what makes the two comparable: any difference is IDF, saturation and
// length normalisation, and nothing else.
func (r *BM25Retriever) Retrieve(ctx context.Context, query string, k int) ([]Result, error) {
	stats, err := r.stats.Stats(ctx)
	if err != nil {
		return nil, err
	}

	terms, err := r.src.QueryTerms(ctx, query)
	if err != nil {
		return nil, err
	}
	if len(terms) == 0 {
		return nil, nil
	}

	rows, err := r.src.MatchingRows(ctx, query, r.matchAll)
	if err != nil {
		return nil, err
	}

	var out []Result
	for _, d := range rows {
		score := BM25Score(stats, d.ID, terms, r.k1, r.b)
		out = append(out, Result{
			ID: d.ID, Text: d.Text, Source: d.Source, Meta: d.Meta,
			Score: score, KeywordScore: score,
		})
	}

	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out, nil
}

// BM25Score is Okapi BM25:
//
//	Σ IDF(t) · ( tf·(k1+1) ) / ( tf + k1·(1 − b + b·len/avgLen) )
//	IDF(t) = ln( 1 + (N − df + 0.5) / (df + 0.5) )
//
// The +1 inside the logarithm is the standard guard that keeps IDF
// non-negative for a term appearing in more than half the corpus, which would
// otherwise let a very common term subtract score.
// It is a pure function over store.Stats rather than a method, so it can be unit
// tested against hand-built statistics with no database at all — and so it
// lands in the retrieval package rather than the storage one when these split.
func BM25Score(s store.Stats, id string, terms []string, k1, b float64) float64 {
	if s.AvgLen == 0 {
		return 0
	}
	docTF, docLen := s.TF[id], s.DocLen[id]

	var total float64
	for _, t := range terms {
		tf := docTF[t]
		if tf == 0 {
			continue
		}
		idf := math.Log(1 + (s.N-s.DF[t]+0.5)/(s.DF[t]+0.5))
		total += idf * (tf * (k1 + 1)) / (tf + k1*(1-b+b*docLen/s.AvgLen))
	}
	return total
}
