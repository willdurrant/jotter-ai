// Package chunk defines what a document and a chunk are, and how one becomes
// the other. It depends on nothing.
package chunk

// Document formats. Format drives preprocessing: PDF text extraction produces
// messy spacing that benefits from collapsing, whereas a text file's line
// structure is real signal the chunker splits on.
const (
	FormatPDF  = "pdf"
	FormatText = "text"
)

// Document is one source document's extracted text. Source identifies where it
// came from, and travels with each chunk as metadata so a retrieval result can
// be attributed back to a document.
//
// How a Document is OBTAINED is deliberately not this module's concern. Reading
// a directory of files is one way and an application's own storage is another,
// and a library that assumed the first would be wrong for most callers.
type Document struct {
	Source string
	Text   string
	Format string
}

// Chunk is a slice of a document, tagged with the document it came from.
type Chunk struct {
	Text   string
	Source string
}

// Chunker turns documents into indexable chunks. The implementations differ in
// how they decide where to cut, which is one of the main things worth measuring.
type Chunker interface {
	ChunkDocuments(docs []Document) ([]Chunk, error)
	Name() string
}
