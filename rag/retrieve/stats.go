package retrieve

import (
	"context"
	"sync"
	"time"

	"github.com/willdurrant/jotter-ai/rag/store"
)

// StatsCache memoises the corpus term statistics, which cost a full scan of
// every lexeme of every row.
//
// It replaces a sync.Once, which was right for a process that indexes once and
// then measures, and wrong for anything long-lived. The Once had two faults,
// and only the first is obvious:
//
//   - It cached the ERROR as well as the value. One transient connection
//     failure at startup disabled BM25 for the life of the process, with no
//     path back.
//   - It never expired. After a reingest every score was computed against a
//     corpus that no longer existed — and nothing about that looks wrong from
//     the outside, because the numbers stay plausible.
//
// A ttl of zero keeps the old caching behaviour, minus the error caching.
type StatsCache struct {
	src store.Lexicon
	ttl time.Duration

	mu       sync.Mutex
	stats    store.Stats
	loadedAt time.Time
	loaded   bool
}

// NewStatsCache caches statistics from src. ttl <= 0 never expires; call
// Invalidate after a reingest instead.
func NewStatsCache(src store.Lexicon, ttl time.Duration) *StatsCache {
	return &StatsCache{src: src, ttl: ttl}
}

// Stats returns the cached statistics, loading them if absent or stale.
//
// An error is never cached: the next call tries again. Concurrent callers are
// serialised rather than deduplicated — the load is rare, and holding the lock
// keeps a reingest from being observed halfway through.
func (c *StatsCache) Stats(ctx context.Context) (store.Stats, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.loaded && (c.ttl <= 0 || time.Since(c.loadedAt) < c.ttl) {
		return c.stats, nil
	}

	s, err := c.src.TermStats(ctx)
	if err != nil {
		return store.Stats{}, err
	}
	c.stats, c.loadedAt, c.loaded = s, time.Now(), true
	return s, nil
}

// Invalidate discards the cache. Call it after writing to the index.
func (c *StatsCache) Invalidate() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.loaded = false
}
