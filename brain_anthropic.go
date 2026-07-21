package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicModel keeps the demo snappy and cheap; swap to
// anthropic.ModelClaudeOpus4_8 for maximum research quality.
const anthropicModel = anthropic.Model("claude-sonnet-5")

// anthropicBrain is the Claude backend. Each call is stateless (no cross-step
// transcript): the agent's memory lives in Jennah, and every prompt is rebuilt
// from freshly-recalled memory, so the reasoner needs no local history.
type anthropicBrain struct {
	client anthropic.Client
}

// newAnthropicBrain builds the Claude backend. When apiKey is non-empty (from
// --anthropic-api-key) it's passed explicitly; otherwise the SDK falls back to its
// usual ANTHROPIC_API_KEY / ambient profile resolution.
func newAnthropicBrain(apiKey string) *anthropicBrain {
	var opts []option.RequestOption
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	return &anthropicBrain{client: anthropic.NewClient(opts...)}
}

func (b *anthropicBrain) label() string { return "anthropic/" + string(anthropicModel) }

// forceTool runs a single message that MUST call the named tool, and returns the
// tool input as raw JSON. Extended thinking is intentionally off — it conflicts
// with forced tool use.
func (b *anthropicBrain) forceTool(ctx context.Context, system, user string, tool anthropic.ToolParam) ([]byte, error) {
	resp, err := b.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:      anthropicModel,
		MaxTokens:  2048,
		System:     []anthropic.TextBlockParam{{Text: system}},
		Messages:   []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(user))},
		Tools:      []anthropic.ToolUnionParam{{OfTool: &tool}},
		ToolChoice: anthropic.ToolChoiceUnionParam{OfTool: &anthropic.ToolChoiceToolParam{Name: tool.Name}},
	})
	if err != nil {
		return nil, err
	}
	for _, blk := range resp.Content {
		if v, ok := blk.AsAny().(anthropic.ToolUseBlock); ok {
			return []byte(v.JSON.Input.Raw()), nil
		}
	}
	return nil, fmt.Errorf("model did not call %s", tool.Name)
}

func (b *anthropicBrain) plan(ctx context.Context, goal string, covered, known []string) (plan, error) {
	tool := anthropic.ToolParam{
		Name:        planTool,
		Description: anthropic.String(planToolDesc),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"stop":        map[string]any{"type": "boolean", "description": planStopDesc},
				"subquestion": map[string]any{"type": "string", "description": planSubDesc},
				"reason":      map[string]any{"type": "string", "description": planWhyDesc},
			},
			Required: []string{"stop"},
		},
	}
	raw, err := b.forceTool(ctx, planSystem(goal, covered, known), "Decide the next step.", tool)
	if err != nil {
		return plan{}, err
	}
	var p plan
	if err := json.Unmarshal(raw, &p); err != nil {
		return plan{}, fmt.Errorf("decode plan: %w", err)
	}
	return p, nil
}

func (b *anthropicBrain) research(ctx context.Context, goal, subquestion string) (findings, error) {
	tool := anthropic.ToolParam{
		Name:        researchTool,
		Description: anthropic.String(researchToolDesc),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"summary": map[string]any{"type": "string", "description": summaryDesc},
				"triples": map[string]any{
					"type":        "array",
					"description": triplesDesc,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"subject":      map[string]any{"type": "string"},
							"subject_type": map[string]any{"type": "string", "description": "coarse category, e.g. 'concept', 'person', 'system', 'protocol'"},
							"relationship": map[string]any{"type": "string", "description": "short verb phrase, e.g. 'is part of', 'depends on'"},
							"object":       map[string]any{"type": "string"},
							"object_type":  map[string]any{"type": "string", "description": "coarse category of the object"},
						},
						"required": []string{"subject", "relationship", "object"},
					},
				},
			},
			Required: []string{"summary", "triples"},
		},
	}
	raw, err := b.forceTool(ctx, researchSystem(goal, subquestion), "Research this subquestion and record findings.", tool)
	if err != nil {
		return findings{}, err
	}
	var f findings
	if err := json.Unmarshal(raw, &f); err != nil {
		return findings{}, fmt.Errorf("decode findings: %w", err)
	}
	return f, nil
}

func (b *anthropicBrain) answer(ctx context.Context, question string, facts, snippets []string) (string, error) {
	resp, err := b.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropicModel,
		MaxTokens: 1536,
		System:    []anthropic.TextBlockParam{{Text: answerSystem(question, facts, snippets)}},
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(question))},
	})
	if err != nil {
		return "", err
	}
	var reply strings.Builder
	for _, blk := range resp.Content {
		if v, ok := blk.AsAny().(anthropic.TextBlock); ok {
			reply.WriteString(v.Text)
		}
	}
	return strings.TrimSpace(reply.String()), nil
}
