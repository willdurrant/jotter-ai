// Package lcsplit adapts langchaingo's text splitters to chunk.Chunker.
package lcsplit

import (
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/textsplitter"

	"github.com/willdurrant/jotter-ai/rag/chunk"
	"github.com/willdurrant/jotter-ai/rag/corpus"
)

// SplitterChunker adapts any langchaingo TextSplitter to the Chunker interface.
type SplitterChunker struct {
	splitter textsplitter.TextSplitter
	name     string
}

func (c *SplitterChunker) Name() string { return c.name }

func (c *SplitterChunker) ChunkDocuments(docs []corpus.Document) ([]chunk.Chunk, error) {
	var chunks []chunk.Chunk
	for _, d := range docs {
		parts, err := c.splitter.SplitText(d.Text)
		if err != nil {
			return nil, fmt.Errorf("split %s: %w", d.Source, err)
		}
		for _, p := range parts {
			// TokenSplitter cuts on token boundaries, which can land mid-way
			// through a multi-byte rune — a "●" (0xe2 0x97 0x8f) sliced after
			// two bytes. Postgres then rejects the insert with
			// "invalid byte sequence for encoding UTF8". Drop the fragments.
			p = strings.ToValidUTF8(p, "")
			if strings.TrimSpace(p) == "" {
				continue
			}
			chunks = append(chunks, chunk.Chunk{Text: p, Source: d.Source})
		}
	}
	if len(chunks) == 0 {
		return nil, fmt.Errorf("no chunks produced from %d documents", len(docs))
	}
	return chunks, nil
}

// NewRecursive splits on a descending list of separators (paragraph,
// line, word, character), cutting at the coarsest boundary that fits.
func NewRecursive(chunkSize, chunkOverlap int, separators ...string) *SplitterChunker {
	opts := []textsplitter.Option{
		textsplitter.WithChunkSize(chunkSize),
		textsplitter.WithChunkOverlap(chunkOverlap),
	}
	name := fmt.Sprintf("recursive/%d/%d", chunkSize, chunkOverlap)
	if len(separators) > 0 {
		opts = append(opts, textsplitter.WithSeparators(separators))
		name += "/custom-sep"
	}
	return &SplitterChunker{splitter: textsplitter.NewRecursiveCharacter(opts...), name: name}
}

// NewToken sizes chunks in tokens rather than characters — the unit the
// embedding model actually consumes. A 500-character chunk of dense prose is a
// very different amount of information from 500 characters of whitespace-heavy
// PDF text.
func NewToken(chunkSize, chunkOverlap int) *SplitterChunker {
	return &SplitterChunker{
		splitter: textsplitter.NewTokenSplitter(
			textsplitter.WithChunkSize(chunkSize),
			textsplitter.WithChunkOverlap(chunkOverlap),
		),
		name: fmt.Sprintf("token/%d/%d", chunkSize, chunkOverlap),
	}
}

// NewMarkdown splits on heading structure. Included for comparison —
// PDF-extracted text has no markdown headings, so it is expected to behave
// close to a plain character split here.
func NewMarkdown(chunkSize, chunkOverlap int) *SplitterChunker {
	return &SplitterChunker{
		splitter: textsplitter.NewMarkdownTextSplitter(
			textsplitter.WithChunkSize(chunkSize),
			textsplitter.WithChunkOverlap(chunkOverlap),
		),
		name: fmt.Sprintf("markdown/%d/%d", chunkSize, chunkOverlap),
	}
}
