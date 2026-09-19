// Package rag holds the measured defaults shared by the retrieval packages.
package rag

// Measured defaults.
//
// These are exported defaults, NOT enforced policy. Nothing in this package
// reads them implicitly: every one is a constructor argument or a caller's
// choice. That is deliberate, and it is what lets a consumer keep its own
// numbers while this package still records what was measured here and why.

const (
	DefaultChunkSize    = 500
	DefaultChunkOverlap = 50
	DefaultTopK         = 3

	// DefaultScoreThreshold is applied at answer time, not during retrieval
	// benchmarking.
	//
	// It is higher than the 0.30 a narrower corpus tolerated, and the reason is
	// worth keeping. 0.30 took false positives to zero for three eras, then
	// stopped: once recipes joined the corpus a near-topic question ("What
	// temperature should I sous vide a steak?") scored 0.3022, just above the
	// line. 0.40 restored zero false positives at no measured cost to recall.
	//
	// The lesson generalises beyond this number: a safe threshold is not a
	// property of the retrieval stack, it is a property of how close off-topic
	// traffic sits to the corpus. A narrow corpus tolerates a low threshold; a
	// broad one does not, because more of the world becomes near-topic. Which
	// is exactly why this is a default and not a floor the package imposes.
	//
	DefaultScoreThreshold = 0.40

	// DefaultCandidateDepth is how many rows each side of a hybrid query
	// contributes before fusion. Deliberately wider than any realistic k, so a
	// chunk ranked poorly by one retriever can still be rescued by the other.
	DefaultCandidateDepth = 50

	// Embedding models under comparison. 3-large is ~5x the price of 3-small
	// and produces 3072-dimension vectors instead of 1536.
	EmbeddingModelSmall = "text-embedding-3-small"
	EmbeddingModelLarge = "text-embedding-3-large"

	DefaultEmbeddingModel = EmbeddingModelSmall

	DimensionsSmall = 1536
	DimensionsLarge = 3072

	DefaultDimensions = DimensionsSmall

	DefaultChatModel = "claude-haiku-4-5"

	// Okapi BM25's two parameters, at their conventional values.
	//
	// k1 controls term-frequency SATURATION: how quickly a repeated term stops
	// adding score. b controls LENGTH NORMALISATION, from 0 (ignore document
	// length) to 1 (divide fully by it). Postgres ts_rank has neither, and no
	// IDF either — which is the whole reason BM25 is worth measuring here.
	DefaultBM25K1 = 1.2
	DefaultBM25B  = 0.75
)

// Dimensions returns the vector width of a known embedding model.
func Dimensions(model string) int {
	if model == EmbeddingModelLarge {
		return DimensionsLarge
	}
	return DimensionsSmall
}
