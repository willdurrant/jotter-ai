package rewrite

import "testing"

func TestLRUEvictsLeastRecentlyUsed(t *testing.T) {
	c := NewLRU(2)
	c.Put("a", "1")
	c.Put("b", "2")
	c.Get("a") // a is now the most recent
	c.Put("c", "3")

	if _, ok := c.Get("b"); ok {
		t.Error("b should have been evicted, it was least recently used")
	}
	for _, k := range []string{"a", "c"} {
		if _, ok := c.Get(k); !ok {
			t.Errorf("%s should still be cached", k)
		}
	}
}

func TestLRUUnboundedKeepsEverything(t *testing.T) {
	c := NewLRU(0)
	for _, k := range []string{"a", "b", "c", "d", "e"} {
		c.Put(k, k)
	}
	for _, k := range []string{"a", "b", "c", "d", "e"} {
		if _, ok := c.Get(k); !ok {
			t.Errorf("max=0 should be unbounded, but %s was evicted", k)
		}
	}
}

func TestLRUOverwriteDoesNotGrow(t *testing.T) {
	c := NewLRU(2)
	c.Put("a", "1")
	c.Put("a", "2")
	c.Put("b", "3")

	if got, _ := c.Get("a"); got != "2" {
		t.Errorf("a = %q, want the overwritten value", got)
	}
	if _, ok := c.Get("b"); !ok {
		t.Error("b should be cached — an overwrite must not count as a new entry")
	}
}

func TestNoCacheNeverHits(t *testing.T) {
	var c Cache = NoCache{}
	c.Put("a", "1")
	if _, ok := c.Get("a"); ok {
		t.Error("NoCache returned a hit")
	}
}
