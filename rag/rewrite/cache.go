package rewrite

import "container/list"

// Cache stores computed rewrites.
//
// Pluggable because the right policy differs by caller. A sweep that asks the
// same questions under a dozen configurations wants every rewrite kept, since
// each miss is a paid model call. A server running for weeks against questions
// its users type wants a bound, because the old map grew forever and nothing
// ever removed an entry.
//
// Eviction can change what a run COSTS. It cannot change what a run MEASURES:
// a miss recomputes the same rewrite from the same prompt and history.
type Cache interface {
	Get(key string) (string, bool)
	Put(key, value string)
}

// NoCache recomputes every time.
type NoCache struct{}

func (NoCache) Get(string) (string, bool) { return "", false }
func (NoCache) Put(string, string)        {}

// lru is a bounded cache with least-recently-used eviction. max <= 0 is
// unbounded, which is the old behaviour and still the right one for a
// short-lived sweep.
type lru struct {
	max   int
	order *list.List
	items map[string]*list.Element
}

type entry struct{ key, value string }

// NewLRU returns a cache holding at most max entries. Pass 0 for unbounded.
func NewLRU(max int) Cache {
	return &lru{max: max, order: list.New(), items: map[string]*list.Element{}}
}

func (c *lru) Get(key string) (string, bool) {
	el, ok := c.items[key]
	if !ok {
		return "", false
	}
	c.order.MoveToFront(el)
	return el.Value.(*entry).value, true
}

func (c *lru) Put(key, value string) {
	if el, ok := c.items[key]; ok {
		el.Value.(*entry).value = value
		c.order.MoveToFront(el)
		return
	}
	c.items[key] = c.order.PushFront(&entry{key, value})

	if c.max > 0 && c.order.Len() > c.max {
		oldest := c.order.Back()
		c.order.Remove(oldest)
		delete(c.items, oldest.Value.(*entry).key)
	}
}
