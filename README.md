# memscout - an autonomous agent that researches and remembers

A small demo **autonomous agent** that consumes Jennah's public memory APIs the
way any external agent would: plain HTTP/JSON through the `jennah-proxy` gateway,
authenticated with a `jennah_sk_` API key. No Jennah server internals are
imported - this is a standalone Go module, so it doubles as a reference for
outside integrators.

Where its sibling **memchat** is *reactive* (a human speaks, it does one
`query → think → commit` turn), memscout is *proactive*: you hand it a goal and
walk away. It self-directs a research loop with **no human in the turn**, using
Jennah as its durable working memory - so the process is stateless and the run
**resumes across restarts**.

It leans on the parts of **unified memory** a chatbot barely touches: the
**execution log as a plan-of-record** (what makes it resumable), a **deep
multi-hop knowledge graph** (not just one-hop facts), and semantic recall as a
**dedup signal**.

## What it does each step

```
-goal "how does the Raft consensus algorithm work"

loop (no human in the turn):
  ┌─ memory:query (log)       ── which subquestions have I already researched? (resume)
  ├─ memory:query (graph 1-hop)── which entities do I already know? (don't retread)
  │
  ├─ brain.plan   ── LLM picks the next subquestion, or decides to STOP
  ├─ memory:query (semantic)  ── "do I already know this?" dedup signal
  ├─ brain.research── LLM answers it → prose summary + (subj)-[rel]->(obj) triples
  └─ memory:commit ── vector chunk (summary) + graph nodes/edges (each subject
                      linked to the topic anchor via COVERS) + an execution-log
                      entry (atomic)
```

Durable, cross-restart memory is just **reusing the same `agent_instance_id`**,
persisted to `memscout-state.json`. Everything the agent has learned lives in
Jennah; the local state file holds nothing but the id. Kill it mid-run, relaunch,
and it reads its own execution log to continue where it stopped. Graph writes are
idempotent server side, so re-researching a topic just converges.

Then inspect or query what it built - straight from memory, no LLM needed for
`-show`:

```
-show                              # print the accumulated knowledge base
-ask "what problem does leader election solve?"   # synthesize an answer from memory
```

## Prerequisites

1. A Jennah API key for an **approved, entitled** enterprise. Mint one after
   logging in (console or `jnh`):
   `POST /v1/apikeys {"label":"memscout"}` → copy the `secret` (shown once).
2. A reasoning model - Anthropic, or Gemini (via **Google AI Studio** with an API
   key, or via **Vertex AI** with a GCP project + ADC).

The reasoning brain is pluggable: only the LLM differs, every Jennah memory call
is identical. `-provider auto` (the default) picks **Anthropic** when an Anthropic
key is configured, otherwise **Gemini**; force it with `-provider gemini|anthropic`.
Within Gemini, Vertex is used when `GOOGLE_GENAI_USE_VERTEXAI=true` or when
`GOOGLE_CLOUD_PROJECT` is set and no Studio key is present.

The agent's home region is chosen at creation with `-region` (or `$JENNAH_REGION`);
it's applied only on first launch, since an agent is pinned to one region for its
lifetime, and empty uses the platform default. List the available regions with
`jnh agents regions`. The target region must have managed embeddings configured
(prod `db0001` / `us-central1` does) - the demo sends plain text and lets the
server embed it.

> **LLM-as-source.** To stay standalone and reproducible, the "research" step
> answers each subquestion from the model's *own* knowledge. A real deployment
> would swap that one call for web search / retrieval / other tool calls - the
> Jennah memory side is unchanged.

## Run

```sh
export JENNAH_API_KEY=jennah_sk_...

# Anthropic:
export ANTHROPIC_API_KEY=sk-ant-...
go run . -goal "how does the Raft consensus algorithm work"   # auto-selects Anthropic

# …or Gemini via Google AI Studio (API key):
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
go run . -goal "the CAP theorem and its trade-offs"

# …or Gemini via Vertex AI (GCP project + ADC, no API key):
gcloud auth application-default login          # once
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1       # optional; defaults to "global"
go run . -goal "how does the Raft consensus algorithm work"

# inspect / query what it has learned (no new research, no LLM for -show):
go run . -show
go run . -ask "what problem does Raft's leader election solve?"

# other knobs:
go run . -goal "…" -max-steps 20          # research budget per run (default 12)
go run . -provider gemini                 # force a provider regardless of which keys are set
go run . -verbose                         # show resume state, dedup hits, commit receipts
go run . -endpoint http://127.0.0.1:8090  # against a local proxy instead
go run . -region us-central1              # pin the agent's home region (or $JENNAH_REGION)

# …or pass the keys as flags instead of env vars:
go run . -jennah-api-key jennah_sk_... -anthropic-api-key sk-ant-...
```

On start it prints the chosen brain, e.g. `reasoning model: anthropic/claude-sonnet-5`
or `reasoning model: gemini/gemini-2.5-flash (vertex:my-proj/us-central1)`.

Give it a goal, watch it research a handful of subquestions and stop on its own
(or hit `-max-steps`), then `-show` the graph it built or `-ask` it a question.
Run it again with the **same** goal to resume; the planner sees what's already
covered (from the log) and pushes into new ground. Delete the state file to start
a fresh knowledge base.

## Notes

- Each provider defaults to a snappy/cheap model (`claude-sonnet-5`,
  `gemini-2.5-flash`); edit `anthropicModel` in `brain_anthropic.go`
  (→ `anthropic.ModelClaudeOpus4_8`) or `geminiModel` in `brain_gemini.go`
  (→ `gemini-2.5-pro`) for max research quality. Backends live behind the `brain`
  interface in `brain.go`.
- The planner (`next_step`) and researcher (`record_findings`) each use a **forced
  tool call** so the model always returns structured output; `-ask` is plain text
  generation. Extended thinking is off on the forced-tool calls (it conflicts with
  forced tool use).
- The knowledge graph is anchored at a stable `topic` node (Label = your goal).
  Each researched subject is linked to it via a `COVERS` edge, so a 1-hop
  traversal enumerates the KB's entities and a 2-hop reads their relationships.
  Entity **type** is stored in each node's `Properties`. Content-hashed ids give
  every entity/relationship a stable identity, so the graph converges instead of
  fragmenting as research repeats.
- `-verbose` surfaces the memory activity live: the resume state (subquestions
  covered / entities known) each step, the dedup hit count before researching,
  and the commit receipt (`log=… vec=… nodes=… edges=…`) after - handy when demoing.
- Fusion (`link:true`) is intentionally not used - it returns Unimplemented.
- Tried-1: "How does the Raft consensus algorithm work?"
- Tried-2: "How does the transformer architecture work in LLMs?"
- Tried-2: "How does the earth came about and the theories around it?"
