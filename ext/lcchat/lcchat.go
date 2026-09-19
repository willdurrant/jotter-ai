// Package lcchat adapts langchaingo's Anthropic client to llm.Chatter.
package lcchat

import (
	"context"
	"fmt"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"

	"github.com/willdurrant/jotter-ai/llm"
)

// anthropicChatter adapts langchaingo's Anthropic client to llm.Chatter.
//
// Everything langchaingo-shaped about answering lives here, so the rest of the
// module stays provider-neutral.
type anthropicChatter struct {
	llm   *anthropic.LLM
	model string
}

// NewAnthropic builds a llm.Chatter backed by langchaingo. The API key is
// read from ANTHROPIC_API_KEY inside langchaingo, exactly as it was when the
// engine and the rewriter each built this for themselves.
func NewAnthropic(model string) (llm.Chatter, error) {
	llm, err := anthropic.New(anthropic.WithModel(model))
	if err != nil {
		return nil, fmt.Errorf("anthropic client: %w", err)
	}
	return &anthropicChatter{llm: llm, model: model}, nil
}

func (c *anthropicChatter) Model() string { return c.model }

func (c *anthropicChatter) Complete(ctx context.Context, msgs []llm.Message, opts ...llm.Option) (llm.Completion, error) {
	var o llm.Options
	for _, apply := range opts {
		apply(&o)
	}

	var call []llms.CallOption
	if o.Model != "" {
		call = append(call, llms.WithModel(o.Model))
	}
	if o.MaxTokens > 0 {
		call = append(call, llms.WithMaxTokens(o.MaxTokens))
	}

	resp, err := c.llm.GenerateContent(ctx, toLangchain(msgs), call...)
	if err != nil {
		return llm.Completion{}, err
	}
	if len(resp.Choices) == 0 {
		return llm.Completion{}, llm.ErrNoResponse
	}

	choice := resp.Choices[0]
	return llm.Completion{
		Text: choice.Content,
		Usage: llm.Usage{
			InputTokens:  tokenCount(choice.GenerationInfo, "InputTokens"),
			OutputTokens: tokenCount(choice.GenerationInfo, "OutputTokens"),
		},
	}, nil
}

// toLangchain maps roles the same way the engine and the rewriter did inline:
// assistant becomes AI, system becomes System, and anything else is a human
// turn. Preserved exactly, because a role change would alter what the model is
// shown and every measured number with it.
func toLangchain(msgs []llm.Message) []llms.MessageContent {
	out := make([]llms.MessageContent, 0, len(msgs))
	for _, m := range msgs {
		role := llms.ChatMessageTypeHuman
		switch m.Role {
		case llm.RoleSystem:
			role = llms.ChatMessageTypeSystem
		case llm.RoleAssistant:
			role = llms.ChatMessageTypeAI
		}
		out = append(out, llms.TextParts(role, m.Content))
	}
	return out
}

// tokenCount reads a usage field langchaingo carries as an untyped any.
func tokenCount(info map[string]any, key string) int {
	switch v := info[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	}
	return 0
}
