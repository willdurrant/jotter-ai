// Package rewrite transforms a question before it is embedded for retrieval.
package rewrite

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/willdurrant/jotter-ai/llm"
)

// Turn is one message of a conversation, oldest first.
//
// An alias rather than its own type: a conversation loaded from a database is
// already a list of role-and-content pairs, and a second identical struct
// would mean every caller converting between them for no reason.
type Turn = llm.Message

// Rewriter transforms a question before it is embedded for retrieval.
//
// The premise: a question and the passage answering it are written in
// different registers. "What type of policy is this?" shares little surface
// vocabulary with "This is a combined buildings and contents policy...",
// so their embeddings sit further apart than the semantics warrant.
//
// history is what makes a FOLLOW-UP question answerable. "Does it include home
// breakdown?" carries a pronoun with no antecedent, so embedded as written it
// retrieves the wrong thing — and a rewriter given no history invents an
// antecedent rather than resolving one, which measured WORSE than not
// rewriting at all. StandaloneQueryPrompt asks for exactly this, so it must be
// given the history it asks about.
//
// Pass nil for a standalone question. Every implementation must then behave as
// it did before history existed, so single-turn measurements stay comparable.
type Rewriter interface {
	Rewrite(ctx context.Context, history []Turn, question string) (string, error)
	Name() string
}

// NoRewrite embeds the question as written — the baseline.
type NoRewrite struct{}

func (NoRewrite) Name() string { return "none (baseline)" }

// NoRewrite ignores history as well as the question, so it stays the true
// baseline: whatever the user typed is what gets embedded.
func (NoRewrite) Rewrite(_ context.Context, _ []Turn, q string) (string, error) {
	return q, nil
}

// Prompts. StandaloneQueryPrompt is the conventional formulation — resolve the
// latest message against the history into one self-contained query.
const (
	StandaloneQueryPrompt = `Given the conversation history below, rewrite the user's latest message into a standalone search query that captures the full intent. The query should work well for semantic search over a note database. Return ONLY the rewritten query, nothing else.

If the latest message is already a clear standalone query, return it unchanged.`

	VocabularyRewritePrompt = `Rewrite the user's question using the formal terminology a policy or contract document would use, so it matches the document's own wording rather than everyday phrasing. Keep it short. Return ONLY the rewritten query, nothing else.`

	HyDEPrompt = `Write a short passage (2-3 sentences) as it would appear in a formal insurance policy document, answering the question below. Invent plausible specifics if needed — factual accuracy does not matter, only that it reads like the source document. Return ONLY the passage.`
)

// LLMRewriter sends the question to Claude with a fixed instruction.
// Rewrites are cached so a comparison over k does not pay for them repeatedly.
type LLMRewriter struct {
	chat   llm.Chatter
	prompt string
	name   string

	mu    sync.Mutex
	cache Cache
}

// Option configures a rewriter.
type Option func(*LLMRewriter)

// WithCache replaces the rewrite cache. The default is NewLRU(1024): bounded,
// because an unbounded map keyed by user input is a leak in anything that runs
// for longer than a test.
func WithCache(c Cache) Option { return func(r *LLMRewriter) { r.cache = c } }

// NewLLMRewriter builds a rewriter. Use StandaloneQueryPrompt to resolve a
// follow-up against its history, VocabularyRewritePrompt to steer toward
// document wording, or HyDEPrompt for hypothetical-document embedding.
func NewLLMRewriter(name, prompt string, chat llm.Chatter, opts ...Option) *LLMRewriter {
	r := &LLMRewriter{
		chat:   chat,
		prompt: prompt,
		name:   name,
		cache:  NewLRU(1024),
	}
	for _, apply := range opts {
		apply(r)
	}
	return r
}

func (r *LLMRewriter) Name() string { return r.name }

func (r *LLMRewriter) Rewrite(ctx context.Context, history []Turn, question string) (string, error) {
	key := cacheKey(history, question)

	r.mu.Lock()
	if cached, ok := r.cache.Get(key); ok {
		r.mu.Unlock()
		return cached, nil
	}
	r.mu.Unlock()

	// With no history this is byte-identical to what it sent before history
	// existed — system prompt, then the bare question — so every single-turn
	// number already measured stays comparable.
	msgs := []llm.Message{{Role: llm.RoleSystem, Content: r.prompt}}
	if len(history) > 0 {
		msgs = append(msgs, llm.Message{Role: llm.RoleUser, Content: formatHistory(history)})
	}
	msgs = append(msgs, llm.Message{Role: llm.RoleUser, Content: question})

	resp, err := r.chat.Complete(ctx, msgs)
	if err != nil {
		if errors.Is(err, llm.ErrNoResponse) {
			return "", fmt.Errorf("empty rewrite for %q", question)
		}
		return "", fmt.Errorf("rewrite %q: %w", question, err)
	}

	out := strings.TrimSpace(resp.Text)
	if out == "" {
		return "", fmt.Errorf("blank rewrite for %q", question)
	}

	r.mu.Lock()
	r.cache.Put(key, out)
	r.mu.Unlock()
	return out, nil
}

// formatHistory renders the conversation as one "role: content" line per turn.
func formatHistory(history []Turn) string {
	var b strings.Builder
	for _, t := range history {
		fmt.Fprintf(&b, "%s: %s\n", t.Role, t.Content)
	}
	return b.String()
}

// cacheKey covers the history as well as the question.
//
// Keyed on the question alone — as it was before history existed — turn two of
// one conversation would be served the rewrite computed for turn two of
// another. "Does it include home breakdown?" resolves to something different
// depending on what was asked first, and the cache would hide that behind a
// plausible answer.
func cacheKey(history []Turn, question string) string {
	if len(history) == 0 {
		return question
	}
	return formatHistory(history) + "\n" + question
}
