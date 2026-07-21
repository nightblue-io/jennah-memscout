// Command memscout is a demo agent: an AUTONOMOUS research/knowledge-base builder
// that self-directs a plan → research → commit loop with NO human in the turn, and
// treats Jennah as its durable working memory. It consumes Jennah's public
// memory APIs exactly the way any external agent would — plain HTTP/JSON
// through the jennah-proxy gateway, authenticated with a jennah_sk_ API key — and
// is a standalone Go module (its own go.mod, not part of the server build) so it
// models a real outside consumer.
//
// memscout is proactive: you hand it a --goal and walk away. It runs a loop until
// it decides the goal is covered (or --max-steps is hit):
//
//  1. LogQuery — read its OWN execution log to see which subquestions it already
//     researched. This is what makes the run resumable: kill it, relaunch, and it
//     picks up where it stopped instead of starting over.
//  2. Graph 1-hop from the stable "topic" anchor — the entities it already knows,
//     so the planner doesn't retread ground.
//  3. brain.plan — the LLM proposes the next subquestion, or decides to stop.
//  4. memory:query (semantic) — "do I already know this?" dedup signal.
//  5. brain.research — the LLM answers the subquestion, returning a prose summary
//     plus (subject)-[relationship]->(object) triples. (LLM-as-source keeps the
//     demo hermetic; a real deployment would swap in web search / tool calls here.)
//  6. memory:commit — the summary as a vector chunk, the triples as graph
//     nodes/edges (each subject also linked to the topic anchor via COVERS), and a
//     turn record to the execution log, atomically.
//
// Cross-session/durable memory is simply reusing the same agent_instance_id: the id
// is persisted to a small state file; everything the agent has learned lives in
// Jennah, so the process is stateless and the loop survives restarts.
//
// The reasoning brain is pluggable (see brain.go): Claude or Gemini depending on
// which API key is present, or an explicit --provider. Only the LLM differs — every
// memory call is identical, which is the whole point of the demo.
//
// Setup (Jennah key + one chat provider). Keys come from env or flags:
//
//	export JENNAH_API_KEY=jennah_sk_...      # from POST /v1/apikeys
//	export ANTHROPIC_API_KEY=sk-ant-...      # Anthropic key, OR
//	export GEMINI_API_KEY=...                # Google AI Studio key
//	go run . -goal "how does the Raft consensus algorithm work"
//
// or inspect / query what it has already learned:
//
//	go run . -show
//	go run . -ask "what problem does Raft's leader election solve?"
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"time"

	agentpb "github.com/alphauslabs/jennah-sdk-go/jennah/agent/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

// verbose makes the memory activity visible on screen: recalled subquestions and
// snippets, and the commit receipt for each research step. Set by --verbose.
var verbose bool

// topicNode is the stable anchor every researched entity hangs off, so a traversal
// from it enumerates the whole knowledge base. Created once (Label = the goal),
// then never re-sent (graph nodes are insert-only server-side).
const topicNode = "topic"

// coversRel links the topic anchor to each top-level entity the agent has covered,
// so a 1-hop traversal lists the KB's entities and a 2-hop reads their relationships.
const coversRel = "COVERS"

func main() {
	var (
		endpoint     = flag.String("endpoint", envOr("JENNAH_ENDPOINT", "https://jennah.alphaus.cloud"), "Jennah proxy origin (http/https)")
		statePath    = flag.String("state", "memscout-state.json", "path to the local state file (just the agent id)")
		provider     = flag.String("provider", "auto", "reasoning LLM: auto|gemini|anthropic (auto prefers Anthropic, else Gemini, by which API key is set)")
		region       = flag.String("region", envOr("JENNAH_REGION", ""), "Jennah home region for the agent (e.g. us-central1); empty uses the platform default. Only applied when creating a new agent workspace. List regions with 'jnh agents regions'")
		jennahKey    = flag.String("jennah-api-key", "", "Jennah API key (jennah_sk_...); falls back to $JENNAH_API_KEY")
		anthropicKey = flag.String("anthropic-api-key", "", "Anthropic API key (sk-ant-...); falls back to $ANTHROPIC_API_KEY")
		goal         = flag.String("goal", "", "research goal/topic; the agent autonomously builds a knowledge base about it")
		maxSteps     = flag.Int("max-steps", 12, "maximum total research steps (across runs) before stopping")
		ask          = flag.String("ask", "", "answer a question from the accumulated memory, then exit (no new research)")
		show         = flag.Bool("show", false, "print the accumulated knowledge base, then exit (no new research)")
	)
	flag.BoolVar(&verbose, "verbose", false, "print recalled memory and commit receipts each step")
	flag.Parse()

	// Fall back to the env vars, but keep them OUT of the flag defaults so --help
	// never prints the actual secret. A flag wins over its env var when both are set.
	*jennahKey = envOr2(*jennahKey, "JENNAH_API_KEY")
	*anthropicKey = envOr2(*anthropicKey, "ANTHROPIC_API_KEY")

	apiKey := *jennahKey
	if apiKey == "" {
		fatal("a Jennah API key is required: pass --jennah-api-key or set JENNAH_API_KEY (a jennah_sk_ key for an approved, entitled enterprise)")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// The one part that varies by provider: the reasoning brain. Everything below is
	// provider-agnostic; the memory APIs don't care which LLM is thinking.
	br, err := newBrain(ctx, *provider, *anthropicKey)
	if err != nil {
		fatal("%v", err)
	}
	fmt.Printf("reasoning model: %s\n", br.label())

	jc := &jennahClient{
		endpoint: strings.TrimRight(*endpoint, "/"),
		token:    apiKey,
		hc:       &http.Client{Timeout: 120 * time.Second},
	}

	st, err := loadState(*statePath)
	if err != nil {
		fatal("load state: %v", err)
	}

	// One-time bootstrap: create the agent workspace and seed the stable "topic"
	// anchor node (Label = the goal), then persist the id. Save only after the seed
	// succeeds, so a crash mid-bootstrap doesn't leave a saved agent without its anchor.
	if st.AgentID == "" {
		if strings.TrimSpace(*goal) == "" {
			fatal("no memory yet: pass -goal \"<topic>\" to start researching")
		}
		id, err := createAgent(ctx, jc, *region)
		if err != nil {
			fatal("create agent: %v", err)
		}
		if _, err := commit(ctx, jc, id, &agentpb.CommitMemoryRequest{
			AgentInstanceId: id,
			Graph:           &agentpb.GraphWrite{Nodes: []*agentpb.GraphNode{{NodeId: topicNode, Label: *goal}}},
		}); err != nil {
			fatal("seed topic node: %v", err)
		}
		st.AgentID = id
		save(*statePath, st)
		if *region != "" {
			fmt.Printf("created agent workspace %s (region %s)\n", id, *region)
		} else {
			fmt.Printf("created agent workspace %s (platform default region)\n", id)
		}
	} else {
		fmt.Printf("reusing agent workspace %s (memory carries over)\n", st.AgentID)
	}

	// Inspection modes: read-only views over the accumulated memory.
	if *show {
		if err := showKB(ctx, jc, st.AgentID); err != nil {
			fatal("show: %v", err)
		}
		return
	}
	if strings.TrimSpace(*ask) != "" {
		if err := answerFromMemory(ctx, jc, br, st.AgentID, *ask); err != nil {
			fatal("ask: %v", err)
		}
		return
	}

	// No goal and an existing workspace: default to a short report of what's known.
	if strings.TrimSpace(*goal) == "" {
		fmt.Println("\n(no -goal given; showing what's already known — pass -goal to research more, -ask to query)")
		if err := showKB(ctx, jc, st.AgentID); err != nil {
			fatal("show: %v", err)
		}
		return
	}

	fmt.Printf("\nmemscout — autonomously researching: %q\n", *goal)
	fmt.Println("(Ctrl-C to stop; progress is saved after every step, so you can resume later.)")
	if err := research(ctx, jc, br, st.AgentID, *goal, *maxSteps); err != nil {
		if errors.Is(err, context.Canceled) {
			fmt.Println("\nstopped — progress is saved in Jennah; rerun to resume.")
			return
		}
		fatal("research: %v", err)
	}
	fmt.Println("\ndone — knowledge saved in Jennah. Try: -show, or -ask \"<question>\".")
}

// research runs the autonomous loop: read the log (resume), consult the planner,
// research one subquestion, commit, repeat — until the planner stops or maxSteps.
func research(ctx context.Context, jc *jennahClient, br brain, agentID, goal string, maxSteps int) error {
	// Progress is global, not per-run: the step number is derived from the number
	// of subquestions already committed to the log (the resume state), so stopping
	// and rerunning continues where it left off instead of restarting at 1.
	prevDone := -1
	for {
		covered, err := coveredSubquestions(ctx, jc, agentID)
		if err != nil {
			return fmt.Errorf("log recall: %w", err)
		}
		known, err := knownEntities(ctx, jc, agentID)
		if err != nil {
			return fmt.Errorf("graph recall: %w", err)
		}
		done := len(covered)
		vlog("resume state: %d subquestion(s) covered, %d entity(ies) known", done, len(known))

		if done >= maxSteps {
			fmt.Printf("\nreached -max-steps (%d); rerun to continue where it left off.\n", maxSteps)
			return nil
		}
		// Backstop: if a committed step failed to advance the covered count (planner
		// repeat, or the log query cap is reached), stop rather than loop forever.
		if done == prevDone {
			fmt.Println("\nno new ground covered; stopping.")
			return nil
		}
		prevDone = done
		step := done + 1

		p, err := br.plan(ctx, goal, covered, known)
		if err != nil {
			return fmt.Errorf("plan: %w", err)
		}
		if p.Stop {
			fmt.Printf("\nplanner: stopping — %s\n", firstNonEmpty(p.Reason, "goal sufficiently covered"))
			return nil
		}
		sq := strings.TrimSpace(p.Subquestion)
		if sq == "" {
			fmt.Println("\nplanner returned no subquestion; stopping.")
			return nil
		}

		// "Do I already know this?" — semantic dedup signal (informational; the
		// server also converges idempotent writes, so a near-duplicate is cheap).
		snips, err := recallSemantic(ctx, jc, agentID, sq, 5)
		if err != nil {
			return fmt.Errorf("semantic recall: %w", err)
		}
		vlog("step %d subquestion: %s", step, sq)
		vlog("  (%d related snippet(s) already in memory)", len(snips))

		f, err := br.research(ctx, goal, sq)
		if err != nil {
			return fmt.Errorf("research step: %w", err)
		}

		resp, err := commitFindings(ctx, jc, agentID, sq, p.Reason, f)
		if err != nil {
			return fmt.Errorf("commit findings: %w", err)
		}
		fmt.Printf("  [%d/%d] %s  →  %d fact(s)\n", step, maxSteps, singleLine(truncate(sq, 80)), len(f.Triples))
		printReceipt(resp)
	}
}

// commitFindings writes one research step atomically: the prose summary as a vector
// chunk, the extracted triples as graph nodes/edges (each subject also linked to the
// topic anchor via COVERS so the KB is enumerable), and a log record that doubles as
// the plan-of-record for resume. Deterministic content-hashed ids give each entity/
// relationship a stable identity so re-researching converges instead of fragmenting.
func commitFindings(ctx context.Context, jc *jennahClient, agentID, subq, reason string, f findings) (*agentpb.CommitMemoryResponse, error) {
	var nodes []*agentpb.GraphNode
	var edges []*agentpb.GraphEdge
	seenN, seenE := map[string]bool{}, map[string]bool{}

	addNode := func(name, typ string) string {
		id := "n_" + hash(strings.ToLower(name))
		if !seenN[id] {
			nodes = append(nodes, &agentpb.GraphNode{NodeId: id, Label: name, Properties: props(typ)})
			seenN[id] = true
		}
		return id
	}
	addEdge := func(id, src, dst, rel string) {
		if id == "" || seenE[id] {
			return
		}
		edges = append(edges, &agentpb.GraphEdge{EdgeId: id, SourceNodeId: src, TargetNodeId: dst, RelationshipType: rel})
		seenE[id] = true
	}

	for _, t := range f.Triples {
		subj := strings.TrimSpace(t.Subject)
		obj := strings.TrimSpace(t.Object)
		if subj == "" || obj == "" {
			continue
		}
		sid := addNode(subj, t.SubjectType)
		oid := addNode(obj, t.ObjectType)
		addEdge("e_"+hash(strings.ToLower(t.Relationship)+"|"+strings.ToLower(subj)+"|"+strings.ToLower(obj)),
			sid, oid, normRel(t.Relationship))
		// Anchor the subject to the topic so a traversal from topic reaches the KB.
		addEdge("e_"+hash("covers|"+strings.ToLower(subj)), topicNode, sid, coversRel)
	}

	req := &agentpb.CommitMemoryRequest{
		AgentInstanceId: agentID,
		Log: &agentpb.ExecutionLogStep{
			StepId:         randID("step"),
			ThoughtProcess: firstNonEmpty(reason, "research step"),
			ToolUsed:       "research",
			ToolInput:      truncate(subq, 500),
			ToolOutput:     truncate(f.Summary, 1000),
		},
		Vectors: []*agentpb.VectorChunk{{
			ChunkId:    randID("chunk"),
			RawContent: "Q: " + subq + "\n" + f.Summary,
		}},
	}
	if len(nodes) > 0 || len(edges) > 0 {
		req.Graph = &agentpb.GraphWrite{Nodes: nodes, Edges: edges}
	}
	return commit(ctx, jc, agentID, req)
}

// ---- read-only views over accumulated memory ----

// showKB prints what the agent has learned: the covered entities (1-hop from topic)
// and the relationships among them (2-hop), read straight back from the graph.
func showKB(ctx context.Context, jc *jennahClient, agentID string) error {
	ents, err := knownEntities(ctx, jc, agentID)
	if err != nil {
		return err
	}
	rels, err := relationships(ctx, jc, agentID)
	if err != nil {
		return err
	}
	covered, err := coveredSubquestions(ctx, jc, agentID)
	if err != nil {
		return err
	}
	fmt.Printf("\nknowledge base — %d entity(ies), %d relationship(s), %d research step(s)\n", len(ents), len(rels), len(covered))
	fmt.Println("\nentities:")
	if len(ents) == 0 {
		fmt.Println("  (none yet)")
	}
	for _, e := range ents {
		fmt.Printf("  · %s\n", e)
	}
	fmt.Println("\nrelationships:")
	if len(rels) == 0 {
		fmt.Println("  (none yet)")
	}
	for _, r := range rels {
		fmt.Printf("  · %s\n", r)
	}
	if len(covered) > 0 {
		fmt.Println("\nresearched subquestions (plan-of-record, newest first):")
		for _, c := range covered {
			fmt.Printf("  · %s\n", singleLine(c))
		}
	}
	return nil
}

// answerFromMemory synthesizes an answer to a question purely from accumulated
// memory: a graph traversal from the topic (facts) plus semantic recall (snippets).
func answerFromMemory(ctx context.Context, jc *jennahClient, br brain, agentID, question string) error {
	rels, err := relationships(ctx, jc, agentID)
	if err != nil {
		return err
	}
	snips, err := recallSemantic(ctx, jc, agentID, question, 8)
	if err != nil {
		return err
	}
	vlog("answering from %d fact(s) and %d snippet(s)", len(rels), len(snips))
	for _, r := range rels {
		vlog("  · %s", r)
	}
	for _, s := range snips {
		vlog("  ~ %s", s)
	}
	if !verbose {
		fmt.Printf("  \033[2m[recalled %d fact(s), %d snippet(s)]\033[0m\n", len(rels), len(snips))
	}
	reply, err := br.answer(ctx, question, rels, snips)
	if err != nil {
		return err
	}
	fmt.Printf("\n%s\n", reply)
	return nil
}

// ---- Jennah memory API calls (HTTP/JSON through the proxy) ----

func recallSemantic(ctx context.Context, jc *jennahClient, agentID, query string, limit int32) ([]string, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Semantic:        &agentpb.SemanticQuery{QueryText: query, Limit: limit},
	}, &resp); err != nil {
		return nil, err
	}
	var out []string
	for _, m := range resp.GetSemantic().GetMatches() {
		if c := strings.TrimSpace(m.GetRawContent()); c != "" {
			out = append(out, singleLine(c))
		}
	}
	return out, nil
}

// coveredSubquestions reads the execution log (newest first) and returns the
// subquestions already researched — the resume/plan-of-record signal.
func coveredSubquestions(ctx context.Context, jc *jennahClient, agentID string) ([]string, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Log:             &agentpb.LogQuery{Limit: 50},
	}, &resp); err != nil {
		return nil, err
	}
	var out []string
	for _, s := range resp.GetLog().GetSteps() {
		if s.GetToolUsed() == "research" {
			if in := strings.TrimSpace(s.GetToolInput()); in != "" {
				out = append(out, in)
			}
		}
	}
	return out, nil
}

// knownEntities lists the top-level entities the agent has covered: a 1-hop
// traversal from the topic anchor along COVERS edges.
func knownEntities(ctx context.Context, jc *jennahClient, agentID string) ([]string, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Graph: &agentpb.GraphQuery{
			Start: &agentpb.GraphNodeMatch{
				Filters: []*agentpb.PropertyFilter{{Key: "NodeId", Value: structpb.NewStringValue(topicNode)}},
			},
			Steps: []*agentpb.GraphStep{{
				Direction:        agentpb.GraphDirection_GRAPH_DIRECTION_OUTGOING,
				RelationshipType: coversRel,
				Node:             &agentpb.GraphNodeMatch{},
			}},
			Limit: 200,
		},
	}, &resp); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var out []string
	for _, row := range resp.GetGraph().GetRows() {
		if v := rowStr(row.AsMap(), "n1_label"); v != "" && !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out, nil
}

// relationships lists the entity-to-entity facts as readable triples: a 2-hop
// traversal (topic -COVERS-> subject -relationship-> object), reading back the
// second hop's edge type and endpoint labels.
func relationships(ctx context.Context, jc *jennahClient, agentID string) ([]string, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Graph: &agentpb.GraphQuery{
			Start: &agentpb.GraphNodeMatch{
				Filters: []*agentpb.PropertyFilter{{Key: "NodeId", Value: structpb.NewStringValue(topicNode)}},
			},
			Steps: []*agentpb.GraphStep{
				{Direction: agentpb.GraphDirection_GRAPH_DIRECTION_OUTGOING, RelationshipType: coversRel, Node: &agentpb.GraphNodeMatch{}},
				{Direction: agentpb.GraphDirection_GRAPH_DIRECTION_OUTGOING, Node: &agentpb.GraphNodeMatch{}},
			},
			Limit: 400,
		},
	}, &resp); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var out []string
	for _, row := range resp.GetGraph().GetRows() {
		m := row.AsMap()
		subj := rowStr(m, "n1_label")
		rel := prettyRel(rowStr(m, "e1_type"))
		obj := rowStr(m, "n2_label")
		if subj == "" || obj == "" {
			continue
		}
		line := fmt.Sprintf("%s %s %s", subj, rel, obj)
		if !seen[line] {
			seen[line] = true
			out = append(out, line)
		}
	}
	return out, nil
}

// createAgent provisions the agent workspace. region is the optional Jennah home
// region ("" = platform default); it's honored only at creation time because an
// agent instance is pinned to one home region for its lifetime.
func createAgent(ctx context.Context, jc *jennahClient, region string) (string, error) {
	id := randID("agent")
	var resp agentpb.CreateAgentResponse
	if _, err := jc.do(ctx, http.MethodPost, "/v1/agents", &agentpb.CreateAgentRequest{
		AgentInstanceId: id,
		AgentName:       "memscout-demo",
		Region:          region,
	}, &resp); err != nil {
		return "", err
	}
	if got := resp.GetAgent().GetAgentInstanceId(); got != "" {
		return got, nil
	}
	return id, nil
}

func commit(ctx context.Context, jc *jennahClient, agentID string, req *agentpb.CommitMemoryRequest) (*agentpb.CommitMemoryResponse, error) {
	var resp agentpb.CommitMemoryResponse
	_, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "commit"), req, &resp)
	return &resp, err
}

func printReceipt(r *agentpb.CommitMemoryResponse) {
	ts := "?"
	if t := r.GetCommitTimestamp(); t != nil {
		ts = t.AsTime().UTC().Format(time.RFC3339)
	}
	vlog("committed: log=%d vec=%d nodes=%d edges=%d @ %s",
		r.GetExecutionLogRows(), r.GetVectorRows(), r.GetGraphNodeRows(), r.GetGraphEdgeRows(), ts)
}

// ---- HTTP client (protojson over the gateway, Bearer auth) ----

type jennahClient struct {
	endpoint string
	token    string
	hc       *http.Client
}

func (c *jennahClient) do(ctx context.Context, method, path string, in, out proto.Message) (int, error) {
	var body io.Reader
	if in != nil {
		b, err := protojson.Marshal(in)
		if err != nil {
			return 0, err
		}
		body = strings.NewReader(string(b))
	}
	req, err := http.NewRequestWithContext(ctx, method, c.endpoint+path, body)
	if err != nil {
		return 0, err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode >= 300 {
		return resp.StatusCode, fmt.Errorf("%s %s: HTTP %d: %s", method, path, resp.StatusCode, gatewayMessage(raw))
	}
	if out != nil {
		if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(raw, out); err != nil {
			return resp.StatusCode, fmt.Errorf("decode %s: %w", path, err)
		}
	}
	return resp.StatusCode, nil
}

func agentPath(id string) string        { return "/v1/agents/" + url.PathEscape(id) }
func memoryPath(id, verb string) string { return agentPath(id) + "/memory:" + verb }

func gatewayMessage(body []byte) string {
	var gw struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(body, &gw); err == nil && strings.TrimSpace(gw.Message) != "" {
		return strings.TrimSpace(gw.Message)
	}
	return strings.TrimSpace(string(body))
}

// ---- local state ----

// state persists only the agent id — that's what makes memory carry across runs.
// No graph id tracking is needed: graph writes are idempotent server side, so
// re-asserting a node/edge across steps just converges.
type state struct {
	AgentID string `json:"agent_id"`
}

// triple is one (subject)-[relationship]->(object) fact the researcher extracted,
// with a coarse type on each endpoint for the graph node Properties.
type triple struct {
	Subject      string `json:"subject"`
	SubjectType  string `json:"subject_type"`
	Relationship string `json:"relationship"`
	Object       string `json:"object"`
	ObjectType   string `json:"object_type"`
}

// plan is the planner's decision for the next loop iteration.
type plan struct {
	Stop        bool   `json:"stop"`
	Subquestion string `json:"subquestion"`
	Reason      string `json:"reason"`
}

// findings is one research step's output: a prose summary (stored as a vector
// chunk) plus the triples that become graph nodes/edges.
type findings struct {
	Summary string   `json:"summary"`
	Triples []triple `json:"triples"`
}

// decodeFirstJSON decodes the first top-level JSON value from b into v, tolerating
// any trailing bytes after it. Some models (or a gateway in front of them) append
// stray prose or markup — e.g. "<...>" — after the tool-call JSON; plain
// json.Unmarshal rejects that as `invalid character '<' after top-level value`,
// whereas a json.Decoder reads exactly one value and stops.
func decodeFirstJSON(b []byte, v any) error {
	return json.NewDecoder(bytes.NewReader(b)).Decode(v)
}

// UnmarshalJSON tolerates models (Gemini in particular) that occasionally emit
// the triples array as a JSON-encoded string ("[{...}]") instead of a real
// array, and models/gateways that append stray bytes after the JSON object. It
// decodes the normal shape first, and on failure falls back to unwrapping a
// string-encoded array.
func (f *findings) UnmarshalJSON(b []byte) error {
	var aux struct {
		Summary string          `json:"summary"`
		Triples json.RawMessage `json:"triples"`
	}
	if err := decodeFirstJSON(b, &aux); err != nil {
		return err
	}
	f.Summary = aux.Summary
	f.Triples = nil
	if len(aux.Triples) == 0 || string(aux.Triples) == "null" {
		return nil
	}
	if err := json.Unmarshal(aux.Triples, &f.Triples); err == nil {
		return nil
	}
	// Fallback: triples came back as a JSON string containing the array.
	var s string
	if err := json.Unmarshal(aux.Triples, &s); err != nil {
		return fmt.Errorf("triples: not an array or string-encoded array")
	}
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return decodeFirstJSON([]byte(s), &f.Triples)
}

func loadState(path string) (*state, error) {
	st := &state{}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return st, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, st); err != nil {
		return nil, err
	}
	return st, nil
}

func save(path string, st *state) {
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: marshal state: %v\n", err)
		return
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "warning: write state %s: %v\n", path, err)
	}
}

// ---- helpers ----

// props builds a GraphNode.Properties struct carrying the entity's coarse type,
// or nil when no type was extracted.
func props(typ string) *structpb.Struct {
	typ = strings.TrimSpace(typ)
	if typ == "" {
		return nil
	}
	return &structpb.Struct{Fields: map[string]*structpb.Value{"type": structpb.NewStringValue(typ)}}
}

func hash(s string) string {
	sum := sha1.Sum([]byte(s))
	return hex.EncodeToString(sum[:])[:12]
}

func randID(prefix string) string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return prefix + "_" + hex.EncodeToString(b[:])
}

// normRel turns a verb phrase into an edge RelationshipType, e.g. "is part of" -> "IS_PART_OF".
func normRel(s string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(strings.TrimSpace(s)) {
		switch {
		case r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "RELATED_TO"
	}
	return out
}

// prettyRel renders a stored RelationshipType back as a readable phrase.
func prettyRel(s string) string {
	if s == "" {
		return "→"
	}
	return strings.ToLower(strings.ReplaceAll(s, "_", " "))
}

func rowStr(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func singleLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// envOr2 returns val when it's non-empty (a flag was passed), otherwise the value
// of env var key. Unlike envOr, the caller-supplied value wins — so an explicit
// flag overrides the env var, and the env var is never a flag default (keeping
// secrets out of --help).
func envOr2(val, key string) string {
	if strings.TrimSpace(val) != "" {
		return val
	}
	return os.Getenv(key)
}

// vlog prints a dim diagnostic line to stdout, only when --verbose is set.
func vlog(format string, args ...any) {
	if verbose {
		fmt.Printf("  \033[2m%s\033[0m\n", fmt.Sprintf(format, args...))
	}
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "memscout: "+format+"\n", args...)
	os.Exit(1)
}
