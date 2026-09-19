// The whole layout rests on one rule: llm/ and rag/ import nothing outside the
// standard library. Break it and a consumer importing retrieval logic silently
// inherits a database driver, a model client or a C toolchain — and finds out
// when their CGO_ENABLED=0 image stops linking, not here.
//
// A comment cannot enforce that and a README cannot either, so this does.
package rag

import (
	"os/exec"
	"strings"
	"testing"
)

const modulePath = "github.com/willdurrant/jotter-ai"

func TestCoreImportsNothingOutsideTheStandardLibrary(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps",
		modulePath+"/llm/...", modulePath+"/rag/...").CombinedOutput()
	if err != nil {
		t.Fatalf("go list -deps: %v\n%s", err, out)
	}

	var foreign []string
	for _, dep := range strings.Fields(string(out)) {
		if strings.HasPrefix(dep, modulePath) {
			continue // our own packages
		}
		// A standard library path has no dot in its first segment.
		if first, _, _ := strings.Cut(dep, "/"); !strings.Contains(first, ".") {
			continue
		}
		foreign = append(foreign, dep)
	}

	if len(foreign) > 0 {
		t.Errorf("llm/ and rag/ must stay dependency-free, but they now import:\n  %s\n\n"+
			"Anything needing a driver, a model client or cgo belongs under ext/.",
			strings.Join(foreign, "\n  "))
	}
}
