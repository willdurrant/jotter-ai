// Package llm is the narrow provider surface every use case in this module is
// written against. It depends on nothing outside the standard library.
package llm

import (
	"context"
	"errors"
)

// Role is who produced a message. The values are the conventional role
// strings, so a conversation loaded from a database needs no translation on
// the way in.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one turn of a conversation sent to a model.
type Message struct {
	Role    Role
	Content string
}

// Usage is reported because a consumer may bill for it. An application that
// accumulates rewrite and generation tokens onto the message it persists would
// be silently broken by a Chatter that returned only a string.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// Completion is one model response.
type Completion struct {
	Text  string
	Usage Usage
}

// Options are the per-call settings every provider supports. Deliberately
// tiny: anything richer would be implementable by one SDK and not another.
type Options struct {
	Model     string
	MaxTokens int
}

type Option func(*Options)

func WithModel(m string) Option  { return func(o *Options) { o.Model = m } }
func WithMaxTokens(n int) Option { return func(o *Options) { o.MaxTokens = n } }

// ErrNoResponse means the provider returned no choices at all.
//
// A sentinel rather than a message, because the two callers word it
// differently — the engine names the model, the rewriter names the question —
// and both wordings predate this interface. Collapsing them into one string
// here would change what a failing run says.
var ErrNoResponse = errors.New("no response from model")

// Chatter completes a conversation.
//
// Deliberately narrow. Tool use, streaming and structured output are exactly
// where the provider SDKs diverge irreconcilably, and tool use in particular is
// a product concern belonging to the application rather than to this seam. What
// is left is the intersection that langchaingo, the OpenAI SDK and the
// Anthropic SDK can all satisfy.
type Chatter interface {
	Complete(ctx context.Context, msgs []Message, opts ...Option) (Completion, error)

	// Model names what answered. Error messages and bench labels both state
	// the configuration that produced a result, and the model is part of it.
	Model() string
}

// Embedder turns text into vectors.
//
// Two methods rather than one: batch embedding is a different API call with
// different cost, and collapsing them would make ingest pay per-chunk request
// overhead on every document.
type Embedder interface {
	EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error)
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}
