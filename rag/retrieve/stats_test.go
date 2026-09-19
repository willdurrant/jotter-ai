package retrieve

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/willdurrant/jotter-ai/rag/store"
)

type countingLexicon struct {
	calls int
	err   error
	avg   float64
}

func (c *countingLexicon) TermStats(context.Context) (store.Stats, error) {
	c.calls++
	if c.err != nil {
		return store.Stats{}, c.err
	}
	return store.Stats{AvgLen: c.avg}, nil
}
func (c *countingLexicon) QueryTerms(context.Context, string) ([]string, error) { return nil, nil }
func (c *countingLexicon) MatchingRows(context.Context, string, bool) ([]store.Row, error) {
	return nil, nil
}

func TestStatsCacheLoadsOnce(t *testing.T) {
	src := &countingLexicon{avg: 10}
	c := NewStatsCache(src, 0)

	for i := 0; i < 3; i++ {
		if _, err := c.Stats(context.Background()); err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}
	if src.calls != 1 {
		t.Errorf("scanned the corpus %d times, want 1", src.calls)
	}
}

// The sync.Once this replaces cached the error too, so one transient failure
// disabled BM25 for the life of the process.
func TestStatsCacheDoesNotCacheErrors(t *testing.T) {
	src := &countingLexicon{err: errors.New("connection refused"), avg: 10}
	c := NewStatsCache(src, 0)

	if _, err := c.Stats(context.Background()); err == nil {
		t.Fatal("expected the first call to fail")
	}
	src.err = nil

	got, err := c.Stats(context.Background())
	if err != nil {
		t.Fatalf("recovery call failed: %v", err)
	}
	if got.AvgLen != 10 {
		t.Errorf("AvgLen = %v, want 10", got.AvgLen)
	}
	if src.calls != 2 {
		t.Errorf("made %d calls, want 2 — the error must not be cached", src.calls)
	}
}

func TestStatsCacheInvalidateForcesAReload(t *testing.T) {
	src := &countingLexicon{avg: 10}
	c := NewStatsCache(src, 0)

	c.Stats(context.Background())
	c.Invalidate()
	c.Stats(context.Background())

	if src.calls != 2 {
		t.Errorf("made %d calls, want 2 — Invalidate should force a reload", src.calls)
	}
}

func TestStatsCacheExpiresOnTTL(t *testing.T) {
	src := &countingLexicon{avg: 10}
	c := NewStatsCache(src, time.Nanosecond)

	c.Stats(context.Background())
	time.Sleep(time.Millisecond)
	c.Stats(context.Background())

	if src.calls != 2 {
		t.Errorf("made %d calls, want 2 — a stale entry should reload", src.calls)
	}
}
