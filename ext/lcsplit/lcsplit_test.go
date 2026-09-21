package lcsplit

import (
	"strings"
	"testing"

	"github.com/willdurrant/jotter-ai/rag/chunk"
)

func TestChunkRespectsSize(t *testing.T) {
	docs := []chunk.Document{{Source: "x.pdf", Text: strings.Repeat("word ", 2000)}}

	for _, size := range []int{100, 500, 1200} {
		chunks, err := NewRecursive(size, 0).ChunkDocuments(docs)
		if err != nil {
			t.Fatalf("chunk(size=%d): %v", size, err)
		}
		for i, c := range chunks {
			if len(c.Text) > size {
				t.Errorf("chunk(size=%d) chunk %d is %d chars, exceeds size", size, i, len(c.Text))
			}
		}
	}
}

func TestChunkOverlapProducesMoreChunks(t *testing.T) {
	docs := []chunk.Document{{Source: "x.pdf", Text: strings.Repeat("alpha beta gamma delta ", 500)}}

	none, err := NewRecursive(200, 0).ChunkDocuments(docs)
	if err != nil {
		t.Fatalf("chunk without overlap: %v", err)
	}
	overlapped, err := NewRecursive(200, 100).ChunkDocuments(docs)
	if err != nil {
		t.Fatalf("chunk with overlap: %v", err)
	}

	if len(overlapped) <= len(none) {
		t.Errorf("overlap=100 gave %d chunks, overlap=0 gave %d — expected more with overlap",
			len(overlapped), len(none))
	}
}

// TestTokenChunkerSizesInTokens shows the conceptual difference: the same
// nominal size means a very different amount of text.
func TestTokenChunkerSizesInTokens(t *testing.T) {
	docs := []chunk.Document{{Source: "x.pdf", Text: strings.Repeat("alpha beta gamma delta ", 500)}}

	byChars, err := NewRecursive(200, 0).ChunkDocuments(docs)
	if err != nil {
		t.Fatalf("recursive: %v", err)
	}
	byTokens, err := NewToken(200, 0).ChunkDocuments(docs)
	if err != nil {
		t.Fatalf("token: %v", err)
	}

	charAvg := avgLen(byChars)
	tokenAvg := avgLen(byTokens)
	t.Logf("size=200: recursive avg %.0f chars/chunk (%d chunks), token avg %.0f chars/chunk (%d chunks)",
		charAvg, len(byChars), tokenAvg, len(byTokens))

	if tokenAvg <= charAvg {
		t.Errorf("expected 200 tokens to hold more characters than 200 chars, got %.0f vs %.0f", tokenAvg, charAvg)
	}
}

func avgLen(chunks []chunk.Chunk) float64 {
	if len(chunks) == 0 {
		return 0
	}
	total := 0
	for _, c := range chunks {
		total += len(c.Text)
	}
	return float64(total) / float64(len(chunks))
}

func TestChunkDocumentsCarriesSource(t *testing.T) {
	docs := []chunk.Document{
		{Source: "a.pdf", Text: strings.Repeat("alpha beta gamma ", 100)},
		{Source: "b.pdf", Text: strings.Repeat("delta epsilon zeta ", 100)},
	}

	chunks, err := NewRecursive(200, 20).ChunkDocuments(docs)
	if err != nil {
		t.Fatalf("chunk documents: %v", err)
	}

	seen := map[string]int{}
	for _, c := range chunks {
		seen[c.Source]++
		if strings.TrimSpace(c.Text) == "" {
			t.Error("produced an empty chunk")
		}
	}
	if seen["a.pdf"] == 0 || seen["b.pdf"] == 0 {
		t.Errorf("expected chunks from both documents, got %v", seen)
	}
}
