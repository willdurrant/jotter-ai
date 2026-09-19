// Pure scoring tests: no database, no keys, no corpus.
package retrieve

import (
	"math"
	"testing"
)

// TestFuseAlphaEndpoints is the correctness anchor for the whole hybrid path:
// alpha=1.0 must rank exactly as pure vector search, and alpha=0.0 exactly as
// pure keyword. If either drifts, every hybrid comparison number is meaningless.

func TestFuseAlphaEndpoints(t *testing.T) {
	cands := []Result{
		{ID: "a", VectorScore: 0.60, KeywordScore: 0.00},
		{ID: "b", VectorScore: 0.55, KeywordScore: 0.30},
		{ID: "c", VectorScore: 0.50, KeywordScore: 0.10},
	}

	if got := ids(FuseAlpha(cands, 1.0, 3)); got != "a,b,c" {
		t.Errorf("alpha=1.0 order = %s, want a,b,c (pure vector)", got)
	}
	if got := ids(FuseAlpha(cands, 0.0, 3)); got != "b,c,a" {
		t.Errorf("alpha=0.0 order = %s, want b,c,a (pure keyword)", got)
	}
}

// TestFuseAlphaRescues is the point of hybrid: a chunk the vector side ranks
// below the winner, but the keyword side ranks first, should climb once
// blended.
//
// Note this needs at least three candidates. With exactly two, min-max
// normalisation maps each side to {0, 1}, so at alpha=0.5 the two scores are
// necessarily tied — an artifact of the normalisation, not of the ranking.
// Real queries fuse DefaultCandidateDepth (50) rows, so the effect does not arise.
func TestFuseAlphaRescues(t *testing.T) {
	cands := []Result{
		{ID: "vec-favourite", VectorScore: 0.62, KeywordScore: 0.00},
		{ID: "kw-favourite", VectorScore: 0.55, KeywordScore: 0.29},
		{ID: "also-ran", VectorScore: 0.45, KeywordScore: 0.05},
	}

	if top := FuseAlpha(cands, 1.0, 1)[0].ID; top != "vec-favourite" {
		t.Fatalf("alpha=1.0 top = %s, want vec-favourite", top)
	}
	if top := FuseAlpha(cands, 0.5, 1)[0].ID; top != "kw-favourite" {
		t.Errorf("alpha=0.5 top = %s, want kw-favourite — blending did not rescue it", top)
	}
}

// TestFuseAlphaTwoCandidateTie pins the artifact above, so nobody later
// "fixes" a tie that is inherent to min-max normalisation.
func TestFuseAlphaTwoCandidateTie(t *testing.T) {
	cands := []Result{
		{ID: "a", VectorScore: 0.62, KeywordScore: 0.00},
		{ID: "b", VectorScore: 0.48, KeywordScore: 0.29},
	}
	got := FuseAlpha(cands, 0.5, 2)
	if got[0].Score != got[1].Score {
		t.Errorf("expected a tie with two candidates at alpha=0.5, got %.4f vs %.4f",
			got[0].Score, got[1].Score)
	}
}

// TestFuseAlphaDegenerate covers the cases that would otherwise produce NaN
// and silently corrupt the ranking: an empty candidate set, a keyword side
// that matched nothing, and every score identical.
func TestFuseAlphaDegenerate(t *testing.T) {
	if got := FuseAlpha(nil, 0.5, 3); len(got) != 0 {
		t.Errorf("empty input returned %d results", len(got))
	}

	noKeyword := []Result{
		{ID: "a", VectorScore: 0.6},
		{ID: "b", VectorScore: 0.4},
	}
	for _, r := range FuseAlpha(noKeyword, 0.5, 2) {
		if math.IsNaN(r.Score) {
			t.Fatalf("%s scored NaN when the keyword side returned nothing", r.ID)
		}
	}
	if got := ids(FuseAlpha(noKeyword, 0.5, 2)); got != "a,b" {
		t.Errorf("with no keyword hits, order = %s, want vector order a,b", got)
	}

	allEqual := []Result{
		{ID: "a", VectorScore: 0.5, KeywordScore: 0.2},
		{ID: "b", VectorScore: 0.5, KeywordScore: 0.2},
	}
	for _, r := range FuseAlpha(allEqual, 0.5, 2) {
		if math.IsNaN(r.Score) || r.Score != 0 {
			t.Errorf("%s scored %v with an all-equal range, want 0", r.ID, r.Score)
		}
	}
}

func TestFuseAlphaTruncates(t *testing.T) {
	cands := []Result{
		{ID: "a", VectorScore: 0.9}, {ID: "b", VectorScore: 0.8},
		{ID: "c", VectorScore: 0.7}, {ID: "d", VectorScore: 0.6},
	}
	if got := FuseAlpha(cands, 1.0, 2); len(got) != 2 {
		t.Errorf("k=2 returned %d results", len(got))
	}
	if got := FuseAlpha(cands, 1.0, 0); len(got) != 4 {
		t.Errorf("k=0 should not truncate, got %d", len(got))
	}
}

func TestNormalise(t *testing.T) {
	cases := []struct{ v, min, max, want float64 }{
		{0.5, 0.0, 1.0, 0.5},
		{1.0, 0.0, 1.0, 1.0},
		{0.0, 0.0, 1.0, 0.0},
		{0.5, 0.5, 0.5, 0.0}, // degenerate range contributes nothing
		{0.5, 1.0, 0.0, 0.0}, // inverted range treated as degenerate
	}
	for _, c := range cases {
		if got := Normalise(c.v, c.min, c.max); got != c.want {
			t.Errorf("Normalise(%v,%v,%v) = %v, want %v", c.v, c.min, c.max, got, c.want)
		}
	}
}

func ids(rs []Result) string {
	out := ""
	for i, r := range rs {
		if i > 0 {
			out += ","
		}
		out += r.ID
	}
	return out
}

// TestRejectedIn covers the Reject semantics: unlike Expect, which one chunk
// satisfies, Reject is violated by ANY chunk in the result set.
