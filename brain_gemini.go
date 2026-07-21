package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"google.golang.org/genai"
)

// geminiModel keeps the demo snappy and cheap; swap to "gemini-2.5-pro" for
// maximum research quality. The same id works on both the AI Studio (API-key) and
// Vertex AI backends.
const geminiModel = "gemini-2.5-flash"

// geminiBrain is the Google Gemini backend. It talks to either AI Studio (an API
// key in GEMINI_API_KEY / GOOGLE_API_KEY) or Vertex AI (GCP project + location +
// Application Default Credentials). Each call is stateless — the agent's memory
// lives in Jennah, not in a local transcript.
type geminiBrain struct {
	client  *genai.Client
	backend string // for the startup banner, e.g. "vertex:my-proj/us-central1"
}

// useVertexAI reports whether to route Gemini through Vertex AI rather than the AI
// Studio API-key path: either explicitly (GOOGLE_GENAI_USE_VERTEXAI=1|true), or
// implicitly when a GCP project is configured and no Studio key is set.
func useVertexAI() bool {
	switch strings.ToLower(os.Getenv("GOOGLE_GENAI_USE_VERTEXAI")) {
	case "1", "true":
		return true
	}
	return os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" &&
		os.Getenv("GOOGLE_CLOUD_PROJECT") != ""
}

func newGeminiBrain(ctx context.Context) (*geminiBrain, error) {
	cc := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
	backend := "ai-studio"
	if useVertexAI() {
		// Vertex uses Application Default Credentials (run once:
		//   gcloud auth application-default login
		// or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key) — no API
		// key. Project/location come from the standard GCP env vars.
		project := os.Getenv("GOOGLE_CLOUD_PROJECT")
		if project == "" {
			return nil, fmt.Errorf("Vertex AI selected but GOOGLE_CLOUD_PROJECT is not set (and run: gcloud auth application-default login)")
		}
		location := envOr("GOOGLE_CLOUD_LOCATION", os.Getenv("GOOGLE_CLOUD_REGION"))
		if location == "" {
			location = "global" // Gemini 2.5 is served on the global endpoint
		}
		cc.Backend = genai.BackendVertexAI
		cc.Project = project
		cc.Location = location
		backend = "vertex:" + project + "/" + location
	}
	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}
	return &geminiBrain{client: client, backend: backend}, nil
}

func (b *geminiBrain) label() string { return "gemini/" + geminiModel + " (" + b.backend + ")" }

// forceTool runs a single generation that MUST call the named function (Mode=ANY
// restricted to that function) and returns its arguments re-marshalled as JSON.
func (b *geminiBrain) forceTool(ctx context.Context, system, user string, decl *genai.FunctionDeclaration) ([]byte, error) {
	config := &genai.GenerateContentConfig{
		MaxOutputTokens:   2048,
		SystemInstruction: genai.NewContentFromText(system, genai.RoleUser),
		Tools:             []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{decl}}},
		ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode:                 genai.FunctionCallingConfigModeAny,
			AllowedFunctionNames: []string{decl.Name},
		}},
	}
	resp, err := b.client.Models.GenerateContent(ctx, geminiModel, []*genai.Content{genai.NewContentFromText(user, genai.RoleUser)}, config)
	if err != nil {
		return nil, err
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return nil, fmt.Errorf("model returned no content")
	}
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.FunctionCall != nil {
			return json.Marshal(part.FunctionCall.Args)
		}
	}
	return nil, fmt.Errorf("model did not call %s", decl.Name)
}

func (b *geminiBrain) plan(ctx context.Context, goal string, covered, known []string) (plan, error) {
	decl := &genai.FunctionDeclaration{
		Name:        planTool,
		Description: planToolDesc,
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"stop":        {Type: genai.TypeBoolean, Description: planStopDesc},
				"subquestion": {Type: genai.TypeString, Description: planSubDesc},
				"reason":      {Type: genai.TypeString, Description: planWhyDesc},
			},
			Required: []string{"stop"},
		},
	}
	raw, err := b.forceTool(ctx, planSystem(goal, covered, known), "Decide the next step.", decl)
	if err != nil {
		return plan{}, err
	}
	var p plan
	if err := json.Unmarshal(raw, &p); err != nil {
		return plan{}, fmt.Errorf("decode plan: %w", err)
	}
	return p, nil
}

func (b *geminiBrain) research(ctx context.Context, goal, subquestion string) (findings, error) {
	decl := &genai.FunctionDeclaration{
		Name:        researchTool,
		Description: researchToolDesc,
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"summary": {Type: genai.TypeString, Description: summaryDesc},
				"triples": {
					Type:        genai.TypeArray,
					Description: triplesDesc,
					Items: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"subject":      {Type: genai.TypeString},
							"subject_type": {Type: genai.TypeString, Description: "coarse category, e.g. 'concept', 'person', 'system', 'protocol'"},
							"relationship": {Type: genai.TypeString, Description: "short verb phrase, e.g. 'is part of', 'depends on'"},
							"object":       {Type: genai.TypeString},
							"object_type":  {Type: genai.TypeString, Description: "coarse category of the object"},
						},
						Required: []string{"subject", "relationship", "object"},
					},
				},
			},
			Required: []string{"summary", "triples"},
		},
	}
	raw, err := b.forceTool(ctx, researchSystem(goal, subquestion), "Research this subquestion and record findings.", decl)
	if err != nil {
		return findings{}, err
	}
	var f findings
	if err := json.Unmarshal(raw, &f); err != nil {
		return findings{}, fmt.Errorf("decode findings: %w", err)
	}
	return f, nil
}

func (b *geminiBrain) answer(ctx context.Context, question string, facts, snippets []string) (string, error) {
	config := &genai.GenerateContentConfig{
		MaxOutputTokens:   1536,
		SystemInstruction: genai.NewContentFromText(answerSystem(question, facts, snippets), genai.RoleUser),
	}
	resp, err := b.client.Models.GenerateContent(ctx, geminiModel, []*genai.Content{genai.NewContentFromText(question, genai.RoleUser)}, config)
	if err != nil {
		return "", err
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return "", fmt.Errorf("model returned no content")
	}
	var reply strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.Text != "" && !part.Thought {
			reply.WriteString(part.Text)
		}
	}
	return strings.TrimSpace(reply.String()), nil
}
