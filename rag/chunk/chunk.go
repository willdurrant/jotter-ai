// Package chunk defines what a chunk is and how documents become chunks.
package chunk

import "github.com/willdurrant/jotter-ai/rag/corpus"

// Chunk is a slice of a document, tagged with the document it came from.
type Chunk struct {
	Text   string
	Source string
}

// Chunker turns loaded documents into indexable chunks. The implementations
// differ in how they decide where to cut, which is one of the main things
// worth measuring.
type Chunker interface {
	ChunkDocuments(docs []corpus.Document) ([]Chunk, error)
	Name() string
}
