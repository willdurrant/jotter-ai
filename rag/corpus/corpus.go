// Package corpus loads source documents from disk and normalises them.
package corpus

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Document formats. Format drives preprocessing: PDF text extraction produces
// messy spacing that benefits from collapsing, whereas a text file's line
// structure is real signal the chunker splits on.
const (
	FormatPDF  = "pdf"
	FormatText = "text"
)

// Document is one source file's extracted text. Source is the path relative
// to the corpus root, and travels with each chunk as metadata so retrieval
// results can be attributed back to a document.
type Document struct {
	Source string
	Text   string
	Format string
}

// formatOf reports the Document format for a path, or "" if the extension is
// not one we ingest.
func formatOf(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".pdf":
		return FormatPDF
	case ".md", ".txt":
		return FormatText
	default:
		return ""
	}
}

// Extractor turns one file into text.
//
// PDF extraction is a plug rather than a dependency. The decoder that actually
// works on real documents is MuPDF, which is cgo — and a consumer that only
// ingests Markdown should not inherit a C toolchain to get it. A service built
// CGO_ENABLED=0 into a distroless image cannot link it at all, so a hard-wired
// import here would be the difference between that image building and not.
type Extractor interface {
	Extract(ctx context.Context, path string) (string, error)
}

type loadOptions struct {
	extractors map[string]Extractor
}

// LoadOption configures a corpus load.
type LoadOption func(*loadOptions)

// WithExtractor registers the extractor for a format. Text formats need none —
// they are read directly — and a format with neither is an error rather than a
// silent omission, because a corpus quietly missing its PDFs measures nothing
// and says so nowhere.
func WithExtractor(format string, e Extractor) LoadOption {
	return func(o *loadOptions) { o.extractors[format] = e }
}

// LoadCorpus reads every PDF, Markdown and plain-text file under dir.
//
// Files named README.md are skipped at any depth: the corpus directory
// documents itself, and that documentation would otherwise be ingested as a
// retrievable document alongside the real content.
func LoadCorpus(ctx context.Context, dir string, opts ...LoadOption) ([]Document, error) {
	o := loadOptions{extractors: map[string]Extractor{}}
	for _, apply := range opts {
		apply(&o)
	}

	var paths []string

	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || formatOf(path) == "" {
			return nil
		}
		if strings.EqualFold(d.Name(), "README.md") {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk corpus %s: %w", dir, err)
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("no documents found under %s", dir)
	}

	// Deterministic order keeps chunk ordering stable between runs.
	sort.Strings(paths)

	docs := make([]Document, 0, len(paths))
	for _, path := range paths {
		format := formatOf(path)

		var text string
		switch e := o.extractors[format]; {
		case e != nil:
			extracted, err := e.Extract(ctx, path)
			if err != nil {
				return nil, err
			}
			text = extracted
		case format == FormatText:
			raw, err := os.ReadFile(path)
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", path, err)
			}
			text = string(raw)
		default:
			return nil, fmt.Errorf("no extractor registered for %s format (%s) — pass WithExtractor(%q, ...)",
				format, path, format)
		}

		rel, err := filepath.Rel(dir, path)
		if err != nil {
			rel = filepath.Base(path)
		}
		docs = append(docs, Document{Source: rel, Text: text, Format: format})
	}
	return docs, nil
}

var whitespaceRun = regexp.MustCompile(`\s+`)

// NormalizeWhitespace collapses runs of whitespace to a single space. PDF
// extraction leaves double spaces between words and newlines mid-sentence
// ("easy  to  \nread,\n \nwrite").
//
// Note this matters less than it looks: langchaingo's embedder defaults to
// StripNewLines = true, so newlines are already removed before embedding.
// What this adds on top is collapsing the double spaces.
func NormalizeWhitespace(text string) string {
	return strings.TrimSpace(whitespaceRun.ReplaceAllString(text, " "))
}

// NormalizeDocuments returns a copy with whitespace collapsed — but only for
// PDFs.
//
// Collapsing is the single most valuable preprocessing step for PDF text,
// worth 17 points of recall , because extraction leaves runs of
// double spaces. It is destructive for text files: it removes every newline,
// and the recursive splitter tries "\n\n" first, so flattening the document
// forces it to cut at arbitrary character positions instead of paragraph
// boundaries. On a recipe it removed all 5 headings and all 63 line breaks.
func NormalizeDocuments(docs []Document) []Document {
	out := make([]Document, len(docs))
	for i, d := range docs {
		out[i] = d
		if d.Format == FormatText {
			continue
		}
		out[i].Text = NormalizeWhitespace(d.Text)
	}
	return out
}
