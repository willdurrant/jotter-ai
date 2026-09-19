// Unit tests for document loading and normalisation. Nothing here touches the
// filesystem beyond asserting that a missing path errors, so these need no
// corpus, no database and no API key.
package corpus

import (
	"context"
	"strings"
	"testing"
)

// TestNormalizeDocumentsPreservesTextStructure pins the per-format rule.
// Collapsing whitespace is worth 17 points of recall on PDFs and is
// destructive on text files: the recursive splitter tries "\n\n" first, so
// flattening a document forces it to cut at arbitrary character positions.
func TestNormalizeDocumentsPreservesTextStructure(t *testing.T) {
	docs := []Document{
		{Source: "a.pdf", Text: "double  spaced\n\nPDF  text", Format: FormatPDF},
		{Source: "b.md", Text: "# Heading\n\n- item one\n- item two", Format: FormatText},
	}

	out := NormalizeDocuments(docs)

	if strings.Contains(out[0].Text, "  ") {
		t.Errorf("PDF text was not collapsed: %q", out[0].Text)
	}
	if out[1].Text != docs[1].Text {
		t.Errorf("text document was modified:\n got %q\nwant %q", out[1].Text, docs[1].Text)
	}
	if !strings.Contains(out[1].Text, "\n") {
		t.Error("text document lost its newlines — the recursive splitter needs them")
	}
}

func TestLoadCorpusMissingDir(t *testing.T) {
	if _, err := LoadCorpus(context.Background(), "does/not/exist"); err == nil {
		t.Fatal("expected an error for a missing corpus directory, got nil")
	}
}

func TestNormalizeWhitespace(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"Python  was  created  in  February  20,1991", "Python was created in February 20,1991"},
		{"easy  to  \nread,\n \nwrite", "easy to read, write"},
		{"  leading and trailing  ", "leading and trailing"},
		{"already normal", "already normal"},
	}

	for _, c := range cases {
		if got := NormalizeWhitespace(c.in); got != c.want {
			t.Errorf("NormalizeWhitespace(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
