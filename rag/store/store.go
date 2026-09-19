// Package store is the seam between retrieval and whatever holds the rows.
package store

import "context"

// It is FOUR small interfaces rather than one, because backends genuinely
// differ in what they can do. A table carrying a pgvector index and no
// full-text index can implement Searcher and nothing else — and that has to
// fail to COMPILE there, rather than fail at runtime in front of a user. A
// single fat interface would force it to stub three methods it cannot honour.
//
// The dividing line is: anything that is arithmetic stays above the seam as a
// pure function; anything that is tokenisation, matching or row retrieval goes
// below it. BM25 falls on both sides — gathering the statistics is SQL, and
// scoring them is not — so it is split rather than pretended to be one thing.

// Row is one stored chunk.
//
// ID is what makes fusion possible: two result lists can only be blended when
// their rows share an identity, which is exactly what langchaingo's
// schema.Document lacks. The pure-vector path goes through langchaingo and so
// still has none; everything that fuses reads its rows directly and does.
type Row struct {
	ID     string
	Text   string
	Source string
}

// Scored is a row with one score. What the number means depends on which call
// produced it — cosine similarity from Search, ts_rank from SearchKeyword —
// and the two are not comparable with each other.
type Scored struct {
	Row
	Score float64
}

// Candidate carries BOTH signals for one row, which is what fusion needs and
// what two independent queries cannot reliably give you.
type Candidate struct {
	Row
	VectorScore  float64
	KeywordScore float64
}

// Query is a similarity search. The store embeds the text rather than the
// retriever, because backends differ in where embedding happens: langchaingo
// embeds inside SimilaritySearch, while an application with its own metered
// embedding call needs to keep the token count that call returns. Putting the
// embedder below the seam leaves both possible.
type Query struct {
	Text string
	K    int
	// MinScore is a similarity floor. Zero means no floor at all, matching
	// what the backends already do — langchaingo omits the predicate entirely
	// rather than comparing against zero.
	MinScore float64
}

// KeywordQuery is a lexical search. MatchAll requires every term to appear;
// see the note on tsqueryAll for why that is the wrong default for RAG.
type KeywordQuery struct {
	Text     string
	K        int
	MatchAll bool
}

// CandidateQuery asks for both signals at once. Depth is how many rows each
// side contributes before fusion, not how many come back.
type CandidateQuery struct {
	Text     string
	Depth    int
	MatchAll bool
}

// Stats is the corpus arithmetic BM25 needs: how long each chunk is, how many
// chunks contain each term, and how long the average chunk is.
//
// Plain maps, because that is exactly what it is, and because anything cleverer
// would put an index in the interface that no backend actually has.
type Stats struct {
	TF     map[string]map[string]float64 // row id -> lexeme -> term frequency
	DocLen map[string]float64            // row id -> total lexemes
	DF     map[string]float64            // lexeme -> rows containing it
	N      float64
	AvgLen float64
}

// Searcher is the ONLY interface a backend must implement to be useful.
type Searcher interface {
	Search(ctx context.Context, q Query) ([]Scored, error)
}

// Lexical needs a full-text index over the same rows the vectors were built
// from. Score is ts_rank.
type Lexical interface {
	SearchKeyword(ctx context.Context, q KeywordQuery) ([]Scored, error)
}

// Fusable returns both signals for the same row identity in one query.
//
// One query rather than two on purpose: the vector side and the keyword side
// have to be joined on row identity, and doing that here means the backend can
// use whatever join its storage makes cheap.
type Fusable interface {
	SearchCandidates(ctx context.Context, q CandidateQuery) ([]Candidate, error)
}

// Lexicon exposes the backend's OWN tokenisation.
//
// This is the part that cannot move above the seam. BM25 is only comparable
// with ts_rank because Postgres tokenises for both, through the same
// to_tsvector('english', ...). A Go tokeniser here would vary two things at
// once and make the difference between them unattributable — which is the
// whole measurement BM25Retriever exists to make.
type Lexicon interface {
	TermStats(ctx context.Context) (Stats, error)
	QueryTerms(ctx context.Context, text string) ([]string, error)
	// MatchingRows returns every row the keyword filter matches, UNSCORED and
	// unlimited: BM25 supplies its own ordering, and the filter must be the
	// identical one SearchKeyword uses so the two differ only in arithmetic.
	MatchingRows(ctx context.Context, text string, matchAll bool) ([]Row, error)
}
