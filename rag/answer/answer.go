// Package answer generates an answer grounded in retrieved context.
package answer

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/willdurrant/jotter-ai/llm"
	"github.com/willdurrant/jotter-ai/rag/retrieve"
	"github.com/willdurrant/jotter-ai/rag/rewrite"
)

// Engine answers questions grounded in retrieved context.
//
// It holds a retrieve.Retriever rather than a *VectorStore so the ANSWER flow can be
// driven by any of the Variables the retrieval comparisons measure — vector,
// keyword or hybrid, with or without a query rewrite. Before that it reached
// into the store directly, which meant every measured alternative was
// unreachable from the flow that generates answers: the comparisons could show
// hybrid winning on a question set and nothing could ask what the model then
// said.
type Engine struct {
	retriever    retrieve.Retriever
	rewriter     rewrite.Rewriter
	chat         llm.Chatter
	topK         int
	instructions Instructions
}

// New takes the retrieval side whole. Ingestion is a separate
// concern, so nothing is embedded here.
//
// The default configuration is NewVector(store, DefaultScoreThreshold)
// with NoRewrite{} — pure vector search, question embedded as written. There is
// no constructor for it: a caller should state its configuration, and a
// convenience wrapper nothing called was one more place for the default to be
// written down and drift.
//
// The similarity floor is a property of the retriever now — NewVector
// holds it, and pass 0 there to disable filtering. Keyword and hybrid retrieval
// have no threshold at all; see the note on retrieve.Result.Score for why one number
// cannot serve all three.
func New(retriever retrieve.Retriever, rewriter rewrite.Rewriter, chat llm.Chatter, topK int) *Engine {
	return &Engine{
		retriever:    retriever,
		rewriter:     rewriter,
		chat:         chat,
		topK:         topK,
		instructions: DefaultInstructions,
	}
}

// WithInstructions replaces the prompt's instruction pair.
//
// It exists so the sentences above the context can be varied while everything
// else — chunks, question, model — is held fixed. That is the only way an
// observed difference is attributable to the instruction rather than to
// anything else, and it is what the findings on Instructions rest on.
func (e *Engine) WithInstructions(in Instructions) *Engine {
	e.instructions = in
	return e
}

// GenerateAnswerWithContext returns the answer and the chunks it was built from.
//
// history is the conversation so far, oldest first, excluding this question.
// Pass nil for a standalone question.
//
// It returns the chunks as well as the answer, and that is not incidental.
// Without them the retrieved context is unobservable, and a wrong answer looks
// identical whether retrieval missed the passage or the model had it in front
// of it and ignored it. Those are different bugs in different layers, and a
// caller that cannot tell them apart cannot fix either.
//
// TWO CHOICES WORTH KNOWING, because a production wiring may well differ:
//
//   - The retrieved chunks go in the USER turn, not the system prompt. Moving
//     them would turn BuildPrompt from one string into a message list, and
//     take with it the ability to show a caller the exact prompt that was sent.
//   - Every turn is rewritten, including the first. An implementation that
//     skips the rewrite when there is no history to resolve against will
//     measure a different path for single-turn questions than for follow-ups.
//
// The rewrite applies to RETRIEVAL only: the model is always shown the question
// as the user asked it. Rewriting exists to move the embedded text closer to
// the document's register, which is a property of the search, not of what the
// user wants answered.
//
// Filtering matters here in a way it does not when measuring recall: without it
// an off-topic question still returns its nearest chunks (scoring ~0.15), and
// the only thing preventing a confident wrong answer is the prompt asking the
// model to say "I don't know". A threshold means genuinely irrelevant context
// never reaches the model at all.
func (e *Engine) GenerateAnswerWithContext(ctx context.Context, history []rewrite.Turn, question string) (string, []retrieve.Result, error) {
	query, err := e.rewriter.Rewrite(ctx, history, question)
	if err != nil {
		return "", nil, err
	}

	results, err := e.retriever.Retrieve(ctx, query, e.topK)
	if err != nil {
		return "", nil, err
	}

	prompt := BuildPrompt(e.instructions, ChunkContext(results), question)

	// System, then the conversation so far, then this turn carrying the
	// retrieved context. With no history that is exactly the two-message call
	// this made before history existed.
	msgs := []llm.Message{{Role: llm.RoleSystem, Content: "You are a helpful assistant"}}
	for _, t := range history {
		role := llm.RoleUser
		if t.Role == "assistant" {
			role = llm.RoleAssistant
		}
		msgs = append(msgs, llm.Message{Role: role, Content: t.Content})
	}
	msgs = append(msgs, llm.Message{Role: llm.RoleUser, Content: prompt})

	out, err := e.chat.Complete(ctx, msgs)
	if err != nil {
		if errors.Is(err, llm.ErrNoResponse) {
			return "", nil, fmt.Errorf("empty response from %s", e.chat.Model())
		}
		return "", nil, fmt.Errorf("generate content: %w", err)
	}

	return out.Text, results, nil
}

// GroundingInstruction is the sentence that makes refusal possible.
//
// Named rather than buried in the template so a caller can assert it actually
// reached the model: drop it and every off-topic question starts being answered
// from the model's own knowledge instead of from the context, which is a
// failure that shows up nowhere near the line that caused it.
const GroundingInstruction = `If the answer is not present in the context, respond with "I don't know"`

// InferenceInstruction permits ONE thing the grounding instruction above
// otherwise forbids: joining two facts that are both in the context.
//
// Measured on "Does Recovery Plus include home breakdown?" with both required
// chunks retrieved. The policy states in Section D that Recovery Plus gets "all
// the benefits of Rescue, Rescue Plus and Recovery", and in Section B that
// Rescue Plus includes home breakdown. That is one inference step, and without
// this line the model refused it — naming the Rescue Plus benefit it had found
// and declining to transfer it.
//
// LOOSENING THE GROUNDING INSTRUCTION IS THE WRONG FIX, and was measured too.
// Replacing it with the softer wording in ProductionGrounding produced a
// confident WRONG answer: "no, Recovery Plus does not include home breakdown",
// having spotted Section B and missed Section D. A refusal is recoverable; a
// wrong answer stated plainly is not.
//
// So the fix is not less grounding but explicit permission to chain WITHIN the
// context. It does not license reaching outside it, which is what the five
// @negative scenarios depend on.
const InferenceInstruction = `Facts stated in different parts of the context may be combined to answer the question.`

// ProductionGrounding is the softer, guidance-style wording that a service
// typically reaches for: it asks the model to stay grounded rather than
// forbidding it to stray. Kept here so the comparison can send the real thing
// rather than a paraphrase — and measured, it answers the two-hop question
// confidently WRONG rather than refusing it.
const ProductionGrounding = `Ground your response in the context provided, and say so honestly if it doesn't contain relevant information.`

// Instructions is the pair of sentences BuildPrompt places above the
// context. Separated from the template so a scenario can vary them and hold
// everything else — chunks, question, model — fixed, which is what makes the
// three outcomes attributable to the instruction rather than to anything else.
//
// An empty inference line is omitted rather than sent blank, so `strict` is
// byte-identical to what shipped before the inference line was added.
type Instructions struct {
	Grounding string
	Inference string
}

var (
	// StrictInstructions refuses anything not literally present — including an
	// answer that has to be derived from two facts that ARE present.
	StrictInstructions = Instructions{Grounding: GroundingInstruction}

	// ProductionInstructions is the softer wording, ungrounded by comparison.
	ProductionInstructions = Instructions{Grounding: ProductionGrounding}

	// DefaultInstructions is what ships: strict, plus permission to chain
	// within the context.
	DefaultInstructions = Instructions{
		Grounding: GroundingInstruction,
		Inference: InferenceInstruction,
	}
)

// block renders the instruction lines, dropping the inference line when unset.
func (in Instructions) block() string {
	if in.Inference == "" {
		return in.Grounding
	}
	return in.Grounding + "\n" + in.Inference
}

// ChunkContext joins retrieved chunks into the context block.
func ChunkContext(results []retrieve.Result) string {
	texts := make([]string, 0, len(results))
	for _, r := range results {
		texts = append(texts, r.Text)
	}
	return strings.Join(texts, "\n\n")
}

// BuildPrompt assembles what is actually sent to the model.
//
// Exported so a caller can SHOW the prompt rather than describe it — useful
// when rendering it with the context block replaced by a summary, since a few
// thousand characters of retrieved text would bury the two things worth
// reading, the instruction and the question. Because the displayed prompt and
// the sent one both go through this function, they cannot drift apart.
func BuildPrompt(in Instructions, contextText, question string) string {
	return fmt.Sprintf(`You are a helpful assistant. Use only the information provided in the context below to answer the question.
%s

Context: %s

question: %s%s`, in.block(), contextText, question, PromptCompletionCue)
}

// PromptCompletionCue is the template's trailing line: the marker telling the
// model where its answer begins. Named rather than written inline so the report
// can trim it back off — it is load-bearing for the model and pure noise to a
// reader — and so the trim cannot drift from the template it is trimming.
const PromptCompletionCue = "\n\nanswer:"
