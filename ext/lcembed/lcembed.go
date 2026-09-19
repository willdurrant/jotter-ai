// Package lcembed builds a langchaingo OpenAI embedder.
//
// It returns langchaingo's own embeddings.Embedder, whose two methods are
// identical to llm.Embedder's, so the result satisfies both and needs no
// wrapper in between.
package lcembed

import (
	"fmt"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"

	"github.com/willdurrant/jotter-ai/llm"
)

// NewOpenAI builds an embedder. There is no embeddings/openai subpackage in
// langchaingo; the embedder is built from an OpenAI LLM client instead.
//
// stripNewLines is exposed because it is worth measuring. It is on by default
// inside langchaingo (embeddings/options.go), which is easy to miss when
// reasoning about how text reaches the model.
func NewOpenAI(model string, stripNewLines bool) (embeddings.Embedder, error) {
	client, err := openai.New(openai.WithEmbeddingModel(model))
	if err != nil {
		return nil, fmt.Errorf("openai client: %w", err)
	}

	embedder, err := embeddings.NewEmbedder(client, embeddings.WithStripNewLines(stripNewLines))
	if err != nil {
		return nil, fmt.Errorf("new embedder: %w", err)
	}
	return embedder, nil
}

// A langchaingo embedder is an llm.Embedder; the interfaces are identical.
var _ func(embeddings.Embedder) llm.Embedder = func(e embeddings.Embedder) llm.Embedder { return e }
