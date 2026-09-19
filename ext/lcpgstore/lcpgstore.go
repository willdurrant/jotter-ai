// Package lcpgstore implements the storage seam over langchaingo's pgvector
// schema. Every SELECT the retrievers need lives here.
package lcpgstore

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgv "github.com/pgvector/pgvector-go"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/pgvector"

	"github.com/willdurrant/jotter-ai/rag"
	"github.com/willdurrant/jotter-ai/rag/chunk"
	"github.com/willdurrant/jotter-ai/rag/store"
)

// Config describes one indexed configuration.
type Config struct {
	Collection string
	// EmbeddingTable must differ per vector width. pgvector types the column
	// as vector(N) and creates the table once, so 1536-wide and 3072-wide
	// embeddings cannot share a table.
	EmbeddingTable string
	Dimensions     int

	// Tracer is installed on every connection this store opens, including the
	// one handed to langchaingo. That last part is why it is a hook rather
	// than something the store does for itself: langchaingo generates the
	// vector statement internally with the threshold INLINED, so it can only
	// be observed at the pgx layer — and observing it is a reporting concern
	// belonging to whoever is producing a report, not to the store.
	//
	// pgx.QueryTracer rather than an interface of our own, so a consumer that
	// already has one composes instead of choosing. Nil means no tracing.
	Tracer pgx.QueryTracer

	// TracerBypass wraps the context used during connection setup, so pgx's
	// own type introspection and statement preparation do not land in a
	// recording. Nil means no bypass.
	TracerBypass func(context.Context) context.Context
}

// Store is a pgvector-backed store for document retrieval.
type Store struct {
	// inner is langchaingo's own store. Unexported: the one thing a caller
	// needs from it is collection removal, which RemoveCollection passes
	// through.
	inner      pgvector.Store
	Collection string

	// Retained so the keyword and hybrid retrievers can query the same rows
	// directly. langchaingo's SimilaritySearch returns schema.Document, which
	// carries no row identity, so fusing two of its result lists would mean
	// matching on document text. Going to SQL avoids that entirely.
	connURL        string
	embeddingTable string
	tracer         pgx.QueryTracer
	bypass         func(context.Context) context.Context
	embedder       embeddings.Embedder

	// conn is the connection langchaingo itself uses. Built here rather than
	// left to pgvector.New so it carries the query tracer — see sqllog.go —
	// which is the only way the vector statement reaches the report as the
	// text that actually ran.
	conn *pgx.Conn

	// pool serves every statement this package issues itself. The retrievers
	// used to dial a fresh connection per query and close it again, which is
	// unremarkable for a bench that runs one query at a time and wrong for a
	// server that runs many.
	pool *pgxpool.Pool
}

// New opens (and creates, if absent) the named collection. Names
// are interpolated into SQL by langchaingo, so keep them to [a-zA-Z0-9_].
func New(ctx context.Context, embedder embeddings.Embedder, connURL string, cfg Config) (*Store, error) {
	if connURL == "" {
		return nil, fmt.Errorf("empty pgvector connection URL (set PGVECTOR_URL)")
	}
	if cfg.Dimensions == 0 {
		cfg.Dimensions = rag.DefaultDimensions
	}

	// Built before the store so pgvector.New receives a traced connection.
	// With a URL it would open exactly this connection itself (pgvector.go:82),
	// so this is behaviour-preserving.
	vs := &Store{
		Collection:     cfg.Collection,
		connURL:        connURL,
		embeddingTable: cfg.EmbeddingTable,
		embedder:       embedder,
		tracer:         cfg.Tracer,
		bypass:         cfg.TracerBypass,
	}
	conn, err := vs.connect(ctx)
	if err != nil {
		return nil, err
	}
	vs.conn = conn

	pool, err := vs.openPool(ctx)
	if err != nil {
		conn.Close(ctx)
		return nil, err
	}
	vs.pool = pool

	opts := []pgvector.Option{
		pgvector.WithConn(conn),
		pgvector.WithEmbedder(embedder),
		pgvector.WithCollectionName(cfg.Collection),
		pgvector.WithVectorDimensions(cfg.Dimensions),
	}
	if cfg.EmbeddingTable != "" {
		opts = append(opts, pgvector.WithEmbeddingTableName(cfg.EmbeddingTable))
	}

	inner, err := pgvector.New(ctx, opts...)
	if err != nil {
		conn.Close(ctx)
		pool.Close()
		return nil, fmt.Errorf("open pgvector collection %q: %w", cfg.Collection, err)
	}
	vs.inner = inner

	// Keyword search reads the same document column, so the index can be added
	// without touching langchaingo's schema or its inserts.
	if err := vs.ensureKeywordIndex(ctx); err != nil {
		conn.Close(ctx)
		pool.Close()
		return nil, err
	}
	return vs, nil
}

// connect opens the single connection langchaingo is given. It carries
// whatever tracer the caller supplied, so anything run on it under a recording
// context lands in the report.
//
// The dial itself runs through the caller's bypass: pgx describes types and
// prepares statements while connecting, and that traffic is not what a reader
// means by "the SQL retrieval used".
func (v *Store) connect(ctx context.Context) (*pgx.Conn, error) {
	cfg, err := pgx.ParseConfig(v.connURL)
	if err != nil {
		return nil, fmt.Errorf("parse connection URL: %w", err)
	}
	if v.tracer != nil {
		cfg.Tracer = v.tracer
	}
	if v.bypass != nil {
		ctx = v.bypass(ctx)
	}

	conn, err := pgx.ConnectConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}
	return conn, nil
}

// openPool builds the pool the direct SQL runs on.
//
// It is pinged here, under the bypass, on purpose. pgxpool connects lazily, so
// without this the FIRST recorded query would also be the one that dials —
// and pgx's connection-time introspection would land in that query's
// attachment, describing traffic the retrieval did not cause.
func (v *Store) openPool(ctx context.Context) (*pgxpool.Pool, error) {
	cfg, err := pgxpool.ParseConfig(v.connURL)
	if err != nil {
		return nil, fmt.Errorf("parse connection URL: %w", err)
	}
	if v.tracer != nil {
		cfg.ConnConfig.Tracer = v.tracer
	}
	if v.bypass != nil {
		ctx = v.bypass(ctx)
	}

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("connect: %w", err)
	}
	return pool, nil
}

// Build embeds the chunks and writes them into the collection, carrying each
// chunk's source document through as metadata.
func (v *Store) Build(ctx context.Context, chunks []chunk.Chunk) error {
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks to index")
	}

	docs := make([]schema.Document, 0, len(chunks))
	for _, c := range chunks {
		docs = append(docs, schema.Document{
			PageContent: c.Text,
			Metadata:    map[string]any{"source": c.Source},
		})
	}

	if _, err := v.inner.AddDocuments(ctx, docs); err != nil {
		return fmt.Errorf("add documents to %q: %w", v.Collection, err)
	}
	return nil
}

// similaritySearch returns the top-k documents. Document.Score is cosine
// similarity (1 - cosine distance), so higher is closer.
func (v *Store) similaritySearch(
	ctx context.Context,
	query string,
	k int,
	opts ...vectorstores.Option,
) ([]schema.Document, error) {
	docs, err := v.inner.SimilaritySearch(ctx, query, k, opts...)
	if err != nil {
		return nil, fmt.Errorf("similarity search in %q: %w", v.Collection, err)
	}
	return docs, nil
}

// Close releases the connection langchaingo was given.
//
// pgvector.Store.Close tests its connection for io.Closer and for a bare
// Close(), and *pgx.Conn — whose signature is Close(ctx) error — satisfies
// neither, so calling it alone is a silent no-op whichever way the store was
// built. Owning the connection is what lets it actually be closed.
func (v *Store) Close() error {
	if err := v.inner.Close(); err != nil {
		return err
	}
	if v.pool != nil {
		v.pool.Close()
	}
	if v.conn != nil {
		return v.conn.Close(context.Background())
	}
	return nil
}

// keywordTable is the physical table holding this collection's rows. Keyword
// and hybrid retrieval query it directly.
func (v *Store) keywordTable() string {
	if v.embeddingTable != "" {
		return v.embeddingTable
	}
	return "langchain_pg_embedding"
}

// ensureKeywordIndex adds a GIN index over the tsvector of the document
// column. It is an expression index, so it adds no column and does not
// disturb langchaingo's inserts.
//
// The explicit 'english' regconfig is required: to_tsvector(regconfig, text)
// is IMMUTABLE and therefore indexable, while the one-argument form is only
// STABLE and Postgres will refuse to index it.
//
// At a few hundred chunks a sequential scan is already instant, so this is
// about modelling the right practice rather than a measurable speed-up.
func (v *Store) ensureKeywordIndex(ctx context.Context) error {
	table := v.keywordTable()
	sql := fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS %s_fts ON %s USING GIN (to_tsvector('english', document))`,
		table, table)
	if _, err := v.pool.Exec(ctx, sql); err != nil {
		return fmt.Errorf("create keyword index on %s: %w", table, err)
	}
	return nil
}

// vectorSearchOption keeps langchaingo's option type out of the Retriever
// interface while still reaching its SimilaritySearch.
type vectorSearchOption = vectorstores.Option

func withThreshold(t float32) vectorSearchOption { return vectorstores.WithScoreThreshold(t) }

// tsqueryAll is websearch_to_tsquery's own output: every term ANDed.
// tsqueryAny rewrites it to OR.
//
// AND is the wrong default for RAG. A natural-language question contains words
// the document never uses — "Which insurer PROVIDES the buildings and contents
// cover?" against a chunk holding insurer/buildings/contents/cover but not
// "provides" fails the conjunction outright and scores nothing, even though it
// is the correct answer. ORing lets ts_rank sort by how many terms matched.
//
// The rewrite runs websearch_to_tsquery first so stemming, stopword removal
// and quoted phrases are handled properly; only the connector changes. An
// all-stopword question yields an empty tsquery, which matches nothing —
// correct, and why a vague question gets no keyword hits at all.
const (
	tsqueryAll = `websearch_to_tsquery('english', %s)`
	tsqueryAny = `replace(websearch_to_tsquery('english', %s)::text, ' & ', ' | ')::tsquery`
)

// tsquery renders the query expression for a given bind parameter.
func tsquery(matchAll bool, param string) string {
	if matchAll {
		return fmt.Sprintf(tsqueryAll, param)
	}
	return fmt.Sprintf(tsqueryAny, param)
}

// --- the storage seam ------------------------------------------------------
//
// Everything below implements Searcher, Lexical, Fusable and Lexicon for
// langchaingo's schema. The SQL is unchanged from when it lived in the
// retrievers; only its address has moved. langchain_pg_collection is still
// written out in full each time, because it is langchaingo's table name and
// this file is the only thing that should know it.

// Search satisfies Searcher.
//
// It delegates to langchaingo rather than issuing its own SQL, deliberately.
// langchaingo inlines the similarity floor as `data.distance < %f` at six
// decimal places rather than binding it, and reproducing that by hand would
// risk moving a number at the boundary for no gain. The cost is that
// schema.Document carries no row identity, so store.Scored.ID is empty on this path
// — exactly as Result.ID was before the seam existed. Nothing fuses these
// rows; everything that does reads them directly below.
func (v *Store) Search(ctx context.Context, q store.Query) ([]store.Scored, error) {
	var opts []vectorstores.Option
	if q.MinScore > 0 {
		opts = append(opts, vectorstores.WithScoreThreshold(float32(q.MinScore)))
	}

	docs, err := v.similaritySearch(ctx, q.Text, q.K, opts...)
	if err != nil {
		return nil, err
	}

	out := make([]store.Scored, 0, len(docs))
	for _, d := range docs {
		out = append(out, store.Scored{
			Row:   store.Row{Text: d.PageContent, Source: sourceOf(d.Metadata)},
			Score: float64(d.Score),
		})
	}
	return out, nil
}

// SearchKeyword satisfies Lexical.
func (v *Store) SearchKeyword(ctx context.Context, kq store.KeywordQuery) ([]store.Scored, error) {
	q := tsquery(kq.MatchAll, "$2")
	sql := fmt.Sprintf(`
SELECT e.uuid::text, e.document, e.cmetadata,
       ts_rank(to_tsvector('english', e.document), %s) AS score
FROM %s e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
WHERE c.name = $1
  AND to_tsvector('english', e.document) @@ %s
ORDER BY score DESC
LIMIT $3`, q, v.keywordTable(), q)

	rows, err := v.pool.Query(ctx, sql, v.Collection, kq.Text, kq.K)
	if err != nil {
		return nil, fmt.Errorf("keyword search: %w", err)
	}
	defer rows.Close()

	var out []store.Scored
	for rows.Next() {
		var id, doc string
		var meta map[string]any
		var score float64
		if err := rows.Scan(&id, &doc, &meta, &score); err != nil {
			return nil, fmt.Errorf("keyword scan: %w", err)
		}
		out = append(out, store.Scored{
			Row:   store.Row{ID: id, Text: doc, Source: sourceOf(meta)},
			Score: score,
		})
	}
	return out, rows.Err()
}

// SearchCandidates satisfies Fusable: one query returning both signals,
// joined on uuid.
func (v *Store) SearchCandidates(ctx context.Context, cq store.CandidateQuery) ([]store.Candidate, error) {
	embedding, err := v.embedder.EmbedQuery(ctx, cq.Text)
	if err != nil {
		return nil, fmt.Errorf("hybrid embed query: %w", err)
	}

	table := v.keywordTable()
	q := tsquery(cq.MatchAll, "$3")
	sql := fmt.Sprintf(`
WITH vec AS (
  SELECT e.uuid, e.document, e.cmetadata,
         1 - (e.embedding <=> $2) AS vscore
  FROM %s e
  JOIN langchain_pg_collection c ON c.uuid = e.collection_id
  WHERE c.name = $1
  ORDER BY e.embedding <=> $2
  LIMIT $4
), kw AS (
  SELECT e.uuid, e.document, e.cmetadata,
         ts_rank(to_tsvector('english', e.document), %s) AS kscore
  FROM %s e
  JOIN langchain_pg_collection c ON c.uuid = e.collection_id
  WHERE c.name = $1
    AND to_tsvector('english', e.document) @@ %s
  ORDER BY kscore DESC
  LIMIT $4
)
SELECT COALESCE(v.uuid, k.uuid)::text,
       COALESCE(v.document, k.document),
       COALESCE(v.cmetadata, k.cmetadata),
       COALESCE(v.vscore, 0), COALESCE(k.kscore, 0)
FROM vec v FULL OUTER JOIN kw k ON v.uuid = k.uuid`, table, q, table, q)

	rows, err := v.pool.Query(ctx, sql, v.Collection, pgv.NewVector(embedding), cq.Text, cq.Depth)
	if err != nil {
		return nil, fmt.Errorf("hybrid search: %w", err)
	}
	defer rows.Close()

	var cands []store.Candidate
	for rows.Next() {
		var id, doc string
		var meta map[string]any
		var vs, kw float64
		if err := rows.Scan(&id, &doc, &meta, &vs, &kw); err != nil {
			return nil, fmt.Errorf("hybrid scan: %w", err)
		}
		cands = append(cands, store.Candidate{
			Row:          store.Row{ID: id, Text: doc, Source: sourceOf(meta)},
			VectorScore:  vs,
			KeywordScore: kw,
		})
	}
	return cands, rows.Err()
}

// TermStats satisfies part of Lexicon. It reads every lexeme of every row once.
//
// unnest(tsvector) returns (lexeme, positions, weights), so term frequency is
// the length of the positions array and no tsvector text has to be parsed.
// Postgres therefore decides tokenisation and stemming for BM25 exactly as it
// does for ts_rank.
func (v *Store) TermStats(ctx context.Context) (store.Stats, error) {
	sql := fmt.Sprintf(`
SELECT e.uuid::text, u.lexeme, coalesce(array_length(u.positions, 1), 1)
FROM %s e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
CROSS JOIN LATERAL unnest(to_tsvector('english', e.document)) u
WHERE c.name = $1`, v.keywordTable())

	rows, err := v.pool.Query(ctx, sql, v.Collection)
	if err != nil {
		return store.Stats{}, fmt.Errorf("bm25 statistics: %w", err)
	}
	defer rows.Close()

	st := store.Stats{
		TF:     map[string]map[string]float64{},
		DocLen: map[string]float64{},
		DF:     map[string]float64{},
	}
	for rows.Next() {
		var id, lexeme string
		var tf float64
		if err := rows.Scan(&id, &lexeme, &tf); err != nil {
			return store.Stats{}, fmt.Errorf("bm25 scan: %w", err)
		}
		if st.TF[id] == nil {
			st.TF[id] = map[string]float64{}
		}
		st.TF[id][lexeme] = tf
		st.DocLen[id] += tf
		st.DF[lexeme]++
	}
	if err := rows.Err(); err != nil {
		return store.Stats{}, err
	}

	st.N = float64(len(st.DocLen))
	var total float64
	for _, l := range st.DocLen {
		total += l
	}
	if st.N > 0 {
		st.AvgLen = total / st.N
	}
	return st, nil
}

// QueryTerms tokenises text the same way the documents were tokenised.
func (v *Store) QueryTerms(ctx context.Context, text string) ([]string, error) {
	rows, err := v.pool.Query(ctx,
		`SELECT lexeme FROM unnest(to_tsvector('english', $1))`, text)
	if err != nil {
		return nil, fmt.Errorf("bm25 query terms: %w", err)
	}
	defer rows.Close()

	var terms []string
	for rows.Next() {
		var t string
		if err := rows.Scan(&t); err != nil {
			return nil, err
		}
		terms = append(terms, t)
	}
	return terms, rows.Err()
}

// MatchingRows satisfies the rest of Lexicon: the identical @@ filter
// SearchKeyword uses, unscored and unlimited.
func (v *Store) MatchingRows(ctx context.Context, text string, matchAll bool) ([]store.Row, error) {
	q := tsquery(matchAll, "$2")
	sql := fmt.Sprintf(`
SELECT e.uuid::text, e.document, e.cmetadata
FROM %s e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
WHERE c.name = $1
  AND to_tsvector('english', e.document) @@ %s`, v.keywordTable(), q)

	rows, err := v.pool.Query(ctx, sql, v.Collection, text)
	if err != nil {
		return nil, fmt.Errorf("bm25 search: %w", err)
	}
	defer rows.Close()

	var out []store.Row
	for rows.Next() {
		var id, doc string
		var meta map[string]any
		if err := rows.Scan(&id, &doc, &meta); err != nil {
			return nil, fmt.Errorf("bm25 scan: %w", err)
		}
		out = append(out, store.Row{ID: id, Text: doc, Source: sourceOf(meta)})
	}
	return out, rows.Err()
}

// sourceOf reads the source document a chunk came from out of langchaingo's
// metadata column. It lives here rather than with the metrics because it is
// knowledge of this schema, and nothing above the seam sees a metadata map.
func sourceOf(metadata map[string]any) string {
	if s, ok := metadata["source"].(string); ok {
		return s
	}
	return ""
}

// RemoveCollection drops the collection and its embeddings.
func (v *Store) RemoveCollection(ctx context.Context, tx pgx.Tx) error {
	return v.inner.RemoveCollection(ctx, tx)
}

var (
	_ store.Searcher = (*Store)(nil)
	_ store.Lexical  = (*Store)(nil)
	_ store.Fusable  = (*Store)(nil)
	_ store.Lexicon  = (*Store)(nil)
)
