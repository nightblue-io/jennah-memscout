package main

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// brain is the pluggable reasoning LLM. Unlike memchat's single chat() call, an
// autonomous researcher needs three distinct capabilities, each provider-agnostic:
//
//   - plan:     decide the next subquestion (or stop) given goal + progress.
//   - research: answer a subquestion, returning a summary + extracted triples.
//   - answer:   synthesize a reply to a user question from recalled memory.
//
// plan and research use a single FORCED tool call so the model always returns
// structured output; answer is plain text. Every Jennah-facing memory call is
// identical regardless of which brain reasons — that's the point of the demo.
type brain interface {
	plan(ctx context.Context, goal string, covered, known []string) (plan, error)
	research(ctx context.Context, goal, subquestion string) (findings, error)
	answer(ctx context.Context, question string, facts, snippets []string) (string, error)
	label() string // short "provider/model" string for the startup banner
}

// Tool definitions, described once and mapped into each SDK's own tool type so the
// two backends stay in lockstep.
const (
	// The planner tool: choose the next research subquestion, or stop.
	planTool     = "next_step"
	planToolDesc = "Decide the single next research subquestion for the goal, given what has already been covered. Set stop=true (and leave subquestion empty) when the goal is sufficiently covered or further questions would be redundant. Otherwise return ONE focused subquestion that advances coverage without repeating covered ground."
	planSubDesc  = "the next focused subquestion to research; empty when stop=true"
	planStopDesc = "true when the goal is sufficiently covered and research should end"
	planWhyDesc  = "one short sentence explaining why this subquestion (or why stopping)"

	// The researcher tool: return a prose summary plus extracted knowledge triples.
	researchTool     = "record_findings"
	researchToolDesc = "Answer the subquestion from your own knowledge and record the findings: a concise prose summary, plus the key facts as (subject)-[relationship]->(object) triples suitable for a knowledge graph. Prefer specific named entities over vague ones; use short verb-phrase relationships (e.g. 'is part of', 'depends on', 'was created by')."
	summaryDesc      = "a concise (2-5 sentence) prose summary answering the subquestion"
	triplesDesc      = "the key facts as subject/relationship/object triples with a coarse type on each entity"
)

// newBrain selects the reasoning provider. "auto" prefers Anthropic when an
// Anthropic key is present, else Gemini — so someone with only one key set just
// runs the command. anthropicKey, when non-empty, is the Anthropic API key from
// --anthropic-api-key (already defaulted to $ANTHROPIC_API_KEY); it overrides the
// SDK's own env lookup.
func newBrain(ctx context.Context, provider, anthropicKey string) (brain, error) {
	if provider == "auto" {
		switch {
		case anthropicKey != "":
			provider = "anthropic"
		case os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "" || useVertexAI():
			provider = "gemini"
		default:
			return nil, fmt.Errorf("no reasoning credentials found: set GEMINI_API_KEY / Vertex AI env (Gemini) or pass --anthropic-api-key / set ANTHROPIC_API_KEY (Anthropic), or pass --provider")
		}
	}
	switch strings.ToLower(provider) {
	case "gemini":
		return newGeminiBrain(ctx)
	case "anthropic", "claude":
		return newAnthropicBrain(anthropicKey), nil
	default:
		return nil, fmt.Errorf("unknown --provider %q (want auto|gemini|anthropic)", provider)
	}
}

// planSystem / researchSystem / answerSystem build the system prompts. They embed
// the freshly-recalled memory so the model reasons over what Jennah already knows.
func planSystem(goal string, covered, known []string) string {
	var b strings.Builder
	b.WriteString("You are memscout, an autonomous research agent building a knowledge base.\n")
	fmt.Fprintf(&b, "Research goal: %s\n\n", goal)
	b.WriteString("# Subquestions already researched\n")
	if len(covered) == 0 {
		b.WriteString("(none yet — this is the first step)\n")
	} else {
		for _, c := range covered {
			b.WriteString("- ")
			b.WriteString(singleLine(c))
			b.WriteString("\n")
		}
	}
	b.WriteString("\n# Entities already in the knowledge base\n")
	if len(known) == 0 {
		b.WriteString("(none yet)\n")
	} else {
		for _, k := range known {
			b.WriteString("- ")
			b.WriteString(k)
			b.WriteString("\n")
		}
	}
	b.WriteString("\nChoose the single most valuable next subquestion, or stop if coverage is sufficient.")
	return b.String()
}

func researchSystem(goal, subquestion string) string {
	return fmt.Sprintf("You are memscout, an autonomous research agent building a knowledge base about: %s\n\n"+
		"Research this subquestion and record your findings: %s\n\n"+
		"Answer accurately from your own knowledge. Extract the salient facts as knowledge-graph triples.",
		goal, subquestion)
}

func answerSystem(question string, facts, snippets []string) string {
	var b strings.Builder
	b.WriteString("You are memscout. Answer the user's question using ONLY the recalled knowledge below, ")
	b.WriteString("which your prior autonomous research committed to long-term memory. ")
	b.WriteString("If the memory doesn't cover it, say so plainly.\n\n")
	fmt.Fprintf(&b, "Question: %s\n\n", question)
	b.WriteString("# Facts (knowledge graph)\n")
	if len(facts) == 0 {
		b.WriteString("(none)\n")
	} else {
		for _, f := range facts {
			b.WriteString("- ")
			b.WriteString(f)
			b.WriteString("\n")
		}
	}
	b.WriteString("\n# Relevant snippets (semantic recall)\n")
	if len(snippets) == 0 {
		b.WriteString("(none)\n")
	} else {
		for _, s := range snippets {
			b.WriteString("- ")
			b.WriteString(s)
			b.WriteString("\n")
		}
	}
	return b.String()
}
