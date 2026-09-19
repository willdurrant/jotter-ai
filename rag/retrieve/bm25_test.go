// BM25Score is pure arithmetic over plain maps, so its three claimed
// properties — IDF, saturation and length normalisation — can be asserted
// without a database. Until this file they were only ever exercised through
// one, which meant the parameters were measured but never actually pinned.
package retrieve

import (
	"testing"

	"github.com/willdurrant/jotter-ai/rag/store"
)

// stats builds a corpus of 100 documents where "rare" appears in one and
// "common" in ninety, with an average length of 10 lexemes.
func stats(docLen float64, tf map[string]float64) store.Stats {
	return store.Stats{
		TF:     map[string]map[string]float64{"d": tf},
		DocLen: map[string]float64{"d": docLen},
		DF:     map[string]float64{"rare": 1, "common": 90},
		N:      100,
		AvgLen: 10,
	}
}

func TestBM25ScoresARareTermFarAboveACommonOne(t *testing.T) {
	s := stats(10, map[string]float64{"rare": 1, "common": 1})

	rare := BM25Score(s, "d", []string{"rare"}, 1.2, 0.75)
	common := BM25Score(s, "d", []string{"common"}, 1.2, 0.75)

	if rare <= common {
		t.Fatalf("rare %.4f should outscore common %.4f — IDF is the whole point", rare, common)
	}
	// ln(1+99.5/1.5) vs ln(1+10.5/90.5): roughly 4.21 against 0.11.
	if ratio := rare / common; ratio < 10 {
		t.Errorf("rare/common = %.2f, expected an order of magnitude", ratio)
	}
}

func TestBM25SaturatesOnTermFrequency(t *testing.T) {
	once := BM25Score(stats(10, map[string]float64{"rare": 1}), "d", []string{"rare"}, 1.2, 0)
	fourTimes := BM25Score(stats(10, map[string]float64{"rare": 4}), "d", []string{"rare"}, 1.2, 0)

	if fourTimes <= once {
		t.Fatalf("four occurrences (%.4f) should beat one (%.4f)", fourTimes, once)
	}
	if ratio := fourTimes / once; ratio >= 4 {
		t.Errorf("four occurrences scored %.2fx — k1 is meant to saturate, not scale linearly", ratio)
	}
}

// k1=0 removes term frequency entirely: presence is all that counts.
func TestBM25WithK1ZeroIgnoresTermFrequency(t *testing.T) {
	once := BM25Score(stats(10, map[string]float64{"rare": 1}), "d", []string{"rare"}, 0, 0)
	fourTimes := BM25Score(stats(10, map[string]float64{"rare": 4}), "d", []string{"rare"}, 0, 0)

	if once != fourTimes {
		t.Errorf("k1=0 should make tf irrelevant, got %.6f and %.6f", once, fourTimes)
	}
}

func TestBM25PenalisesLongDocumentsOnlyWhenBIsSet(t *testing.T) {
	short := BM25Score(stats(5, map[string]float64{"rare": 1}), "d", []string{"rare"}, 1.2, 1)
	long := BM25Score(stats(20, map[string]float64{"rare": 1}), "d", []string{"rare"}, 1.2, 1)
	if short <= long {
		t.Errorf("b=1 should favour the shorter document, got short %.4f long %.4f", short, long)
	}

	shortNoNorm := BM25Score(stats(5, map[string]float64{"rare": 1}), "d", []string{"rare"}, 1.2, 0)
	longNoNorm := BM25Score(stats(20, map[string]float64{"rare": 1}), "d", []string{"rare"}, 1.2, 0)
	if shortNoNorm != longNoNorm {
		t.Errorf("b=0 should ignore length, got %.6f and %.6f", shortNoNorm, longNoNorm)
	}
}

func TestBM25IgnoresTermsTheDocumentDoesNotContain(t *testing.T) {
	s := stats(10, map[string]float64{"rare": 1})
	only := BM25Score(s, "d", []string{"rare"}, 1.2, 0.75)
	plusAbsent := BM25Score(s, "d", []string{"rare", "absent"}, 1.2, 0.75)

	if only != plusAbsent {
		t.Errorf("an absent term changed the score: %.6f then %.6f", only, plusAbsent)
	}
}

// An empty corpus must not divide by zero.
func TestBM25OnEmptyStatsIsZero(t *testing.T) {
	if got := BM25Score(store.Stats{}, "d", []string{"rare"}, 1.2, 0.75); got != 0 {
		t.Errorf("empty statistics scored %.6f, want 0", got)
	}
}
