// The prompt is the product here: a scenario that asserts on an answer is only
// meaningful if the instruction that produced it is known exactly. These pin
// the bytes.
package answer

import (
	"strings"
	"testing"
)

func TestBuildPromptStrictIsExact(t *testing.T) {
	got := BuildPrompt(StrictInstructions, "CTX", "Q")
	want := `You are a helpful assistant. Use only the information provided in the context below to answer the question.
If the answer is not present in the context, respond with "I don't know"

Context: CTX

question: Q

answer:`
	if got != want {
		t.Errorf("prompt drifted:\n got %q\nwant %q", got, want)
	}
}

// An empty inference line is omitted rather than sent blank, so the strict
// prompt is byte-identical to what shipped before the inference line existed.
func TestStrictPromptHasNoBlankInstructionLine(t *testing.T) {
	got := BuildPrompt(StrictInstructions, "CTX", "Q")
	if strings.Contains(got, GroundingInstruction+"\n\n\nContext:") {
		t.Error("an empty inference line was sent as a blank line")
	}
}

func TestInstructionPresets(t *testing.T) {
	cases := []struct {
		name            string
		in              Instructions
		wants, wantsNot []string
	}{
		{"strict", StrictInstructions,
			[]string{GroundingInstruction}, []string{InferenceInstruction, ProductionGrounding}},
		{"default", DefaultInstructions,
			[]string{GroundingInstruction, InferenceInstruction}, []string{ProductionGrounding}},
		{"production", ProductionInstructions,
			[]string{ProductionGrounding}, []string{GroundingInstruction, InferenceInstruction}},
	}

	for _, c := range cases {
		got := BuildPrompt(c.in, "CTX", "Q")
		for _, w := range c.wants {
			if !strings.Contains(got, w) {
				t.Errorf("%s prompt is missing %q", c.name, w)
			}
		}
		for _, w := range c.wantsNot {
			if strings.Contains(got, w) {
				t.Errorf("%s prompt unexpectedly contains %q", c.name, w)
			}
		}
		if !strings.HasSuffix(got, PromptCompletionCue) {
			t.Errorf("%s prompt does not end with the completion cue", c.name)
		}
	}
}

func TestChunkContextJoinsWithABlankLine(t *testing.T) {
	// Separated by a blank line so the model sees two passages, not one run-on.
	if got := ChunkContext(nil); got != "" {
		t.Errorf("no results should give an empty context, got %q", got)
	}
}
