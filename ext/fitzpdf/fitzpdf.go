// Package fitzpdf extracts PDF text with MuPDF. It is the only package in this
// module that needs cgo, and nothing imports it unless a caller asks for it.
package fitzpdf

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/gen2brain/go-fitz"
)

// Loader extracts text from a PDF using MuPDF.
//
// It deliberately does NOT use langchaingo's documentloaders.NewPDF. That is
// backed by github.com/ledongthuc/pdf, which cannot decode embedded subset
// fonts: on a real insurance policy every bolded defined term came out as
// mojibake ("Lead Policyholder" -> "s3I1ocp745oc3 t", "Policy Schedule" ->
// "1ocp74I975 3Bc", 49 occurrences destroyed). Those terms are exactly the
// vocabulary the document is written in, so retrieval had nothing to match.
//
// MuPDF decodes the same file cleanly with zero replacement characters.
// The cost is cgo — builds require CGO_ENABLED=1.
type Loader struct {
	path string
}

func New(path string) (*Loader, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("PDF not found at %s: %w", path, err)
	}
	return &Loader{path: path}, nil
}

// Load returns the whole document as a single string, pages joined by newline.
func (l *Loader) Load(ctx context.Context) (string, error) {
	pages, err := l.LoadPages(ctx)
	if err != nil {
		return "", err
	}
	return strings.Join(pages, "\n"), nil
}

// LoadPages returns the text of each page separately.
func (l *Loader) LoadPages(_ context.Context) ([]string, error) {
	doc, err := fitz.New(l.path)
	if err != nil {
		return nil, fmt.Errorf("open pdf %s: %w", l.path, err)
	}
	defer doc.Close()

	n := doc.NumPage()
	pages := make([]string, 0, n)
	for i := 0; i < n; i++ {
		text, err := doc.Text(i)
		if err != nil {
			return nil, fmt.Errorf("extract page %d of %s: %w", i+1, l.path, err)
		}
		pages = append(pages, text)
	}
	return pages, nil
}

// Extractor adapts Loader to a corpus loader's Extractor interface.
//
// It is the piece that carries the cgo dependency, so registering it is a
// deliberate act by the caller rather than something LoadCorpus does for them.
type Extractor struct{}

func (Extractor) Extract(ctx context.Context, path string) (string, error) {
	loader, err := New(path)
	if err != nil {
		return "", err
	}
	return loader.Load(ctx)
}
