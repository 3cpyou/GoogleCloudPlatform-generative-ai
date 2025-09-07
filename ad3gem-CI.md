AD3Gem Conversational Intelligence — PRD (What It Is / Should Be / Should Do)

Paste this into Cursor. It’s a complete behavior spec — no “how-to”, only contracts, expectations, and outcomes.

⸻

0) One-glance Summary
	•	Firestore(ad3sam-database) = source of live truth (e.g., Gmail-derived records, warehouse facts).
	•	ad3gem-conversation = append-only, turn-by-turn chat history.
	•	ad3gem-memory = refined, categorized, versioned “current truths” + full history.
	•	Background Refiner = periodically distills chat into immutable history events and updates the current belief pointers.
	•	Runtime Bot = answers in natural language; plans each turn; uses recent chat, memory heads, and (when needed) Firestore(ad3sam-database); adapts tactically on follow-ups.

⸻

1) Goals & Non-Goals

Goals
	•	Persist and evolve knowledge from chats without losing history.
	•	Answer naturally, dynamically, and contextually across sessions.
	•	Keep small/simple at first; add structure only when scale demands.
	•	Never delete knowledge; supersede with versions and retain provenance.

Non-Goals
	•	No implementation specifics, code, libraries, or infra recipes here.
	•	Not a general data warehouse design; only the contract this system expects.

⸻

2) System Components (What They ARE)
	1.	Firestore(ad3sam-database) (Warehouse Truth)
	•	Holds authoritative, queryable organizational records (e.g., Gmail messages/threads/attachments, other ops data).
	•	Exposed to the bot via tenant-scoped views.
	2.	ad3gem-conversation (Chat History)
	•	Stores raw conversation turns (user, assistant, tools) with timestamps and minimal metadata.
	•	Append-only; acts as short-term context and the refiner’s source.
	3.	ad3gem-memory (Refined Knowledge)
	•	Heads: tiny docs representing the current belief/value per topic.
	•	Claims: immutable history of extractions, updates, reclassifications.
	•	Optional lightweight taxonomy/aliases to normalize evolving labels.
	4.	Background Refiner
	•	Scheduled process that ingests new turns, extracts atomic facts/preferences/policies, writes new claims, and moves heads.
	5.	Runtime Assistant
	•	Per turn: detects need for warehouse data; assembles minimal context; queries Firestore(ad3sam-database) if required; composes a natural-language answer; logs the turn.

⸻

3) Behavioral Contracts (What It SHOULD DO)

3.1 Answering Behavior (Per Turn)
	•	Plan: Determine if the question needs warehouse data, memory only, or both.
	•	Assemble Context:
	•	Load recent N turns from ad3gem-conversation for dialogue continuity.
	•	Load relevant heads from ad3gem-memory for policies, definitions, preferences, and current beliefs.
	•	Retrieve Live Data (when needed):
	•	Query Firestore(ad3sam-database) through tenant-scoped views with parameters derived from the user ask and remembered defaults.
	•	Compose:
	•	Integrate Firestore(ad3sam-database) results + memory heads + recent turns.
	•	Always respond in natural language, adjusting tone/conciseness to remembered preferences and current cues.
	•	If the user pivots, change tactic (e.g., stop querying, or add a new query) without losing conversational flow.
	•	Log:
	•	Append the assistant’s final message into ad3gem-conversation.

3.2 Memory Evolution (Background)
	•	Ingest: Periodically scan new conversation turns since last checkpoint.
	•	Extract: Identify atomic statements (facts, preferences, boundaries, tools, roles, policies), summaries, corrections, and reclassifications.
	•	Append: Write claims (immutable events) with timestamps and provenance (conversation/message refs).
	•	Supersede: Update heads to point to the latest valid claim when a value changes; do not edit prior claims.
	•	Normalize: Maintain aliases/tags so that future retrieval stays stable as wording evolves.
	•	Scale Gently:
	•	Start flat (e.g., facet=location with city entries).
	•	Promote grouping (e.g., region) only when needed (see §6).

3.3 Conflict & Freshness Policy
	•	If Firestore(ad3sam-database) contradicts a head, prefer Firestore(ad3sam-database) for the reply and mark memory for review/update on the next refinement pass.
	•	Heads are used at runtime; claims are for audit/time-travel and refiner decisions, not for prompt stuffing.

⸻

4) Data Contracts (Names & Required Fields — WHAT exists, not HOW)

These are behavioral contracts for what must be present and queryable. Exact physical layout is chosen by the implementing agent, but names/fields below must be representable.

4.1 ad3gem-conversation

Entity: Conversation
	•	conversationId (id)
	•	orgId
	•	participants (list of user/bot identifiers)
	•	createdAt
	•	lastMessageAt
	•	status (“open” | “closed”)

Entity: Message (child of a conversation)
	•	messageId (id)
	•	role (“user” | “assistant” | “tool”)
	•	text (exact content)
	•	turnIndex (monotonic)
	•	createdAt
	•	Optional: channel, attachments[], meta{}

Contractual behaviors
	•	Messages are append-only and ordered by createdAt / turnIndex.
	•	Recent K messages fetchable per conversation for runtime context.
	•	Globally readable by the refiner via a query across all messages (scoped to org/user).

4.2 ad3gem-memory

Entity: Head (current belief pointer)
	•	headId (dedupe key, e.g., hours:location=yolkrest)
	•	facet (e.g., hours, payments, people, process, tool, boundary, etc.)
	•	scope (e.g., org, location:<slug>, person:<slug>, etc.)
	•	value (typed map or compact text — the current truth)
	•	version (int, increments on change)
	•	tokens[] (normalized labels/dimensions used for retrieval)
	•	firstObservedAt, lastChangedAt, updatedAt
	•	lastDerivedFrom[] (provenance refs to conversations or warehouse items)
	•	status (“active” | “disputed” | “retired”)

Entity: Claim (immutable history)
	•	claimId (id)
	•	kind (“memory” | “belief” | “reclassify”)
	•	facet
	•	scope
	•	statement (text) or value (typed)
	•	confidence (0–1)
	•	salience (0–1)
	•	observedAt (when found in conversation)
	•	ingestedAt (when written as a claim)
	•	sources[] (conversationId/messageId, optional warehouse ref)
	•	supersedesClaimId (nullable)
	•	Optional: aliases[], tags[]

Contractual behaviors
	•	Heads are the only runtime source of persistent knowledge injected into prompts.
	•	Claims are never edited or deleted; supersession creates a new claim and updates the head pointer.
	•	Heads/claims can be filtered by facet, scope, tokens, and recency fields.

4.3 Firestore(ad3sam-database) (Warehouse Truth)

Accessible Entities (examples)
	•	Messages, threads, attachments, or other operational tables, each row-scoped to an organization and time-partitionable.
	•	Exposed to the assistant through tenant-scoped views (names stable, schema abstracted).

Contractual behaviors
	•	The assistant can request tabular results for specific questions.
	•	Results are incorporated into the natural-language answer and may be referenced by view name (human-readable citation), not raw table internals.

⸻

5) Runtime Decisioning (What the Assistant MUST DO)
	•	Natural language in/out: Always speak plainly; keep jargon minimal unless user prefers otherwise.
	•	Dynamic tactic shift: On follow-ups, the assistant re-plans: may drop a query, run a new query, or rely on memory only.
	•	Context budget discipline: Use only the few most recent turns and the few heads relevant to the ask.
	•	Answer precedence:
	1.	Warehouse facts (if queried successfully)
	2.	Current heads (policies, definitions, preferences)
	3.	Recent conversation turns
	4.	Model’s general knowledge (used carefully; align with 1–3)
	•	If nothing is found:
	•	Say so clearly, propose next step (e.g., refine filters, confirm entity, expand date range), and proceed naturally.

⸻

6) Scale & Promotion Rules (When It SHOULD Grow)
	•	Start flat: single-level facets (e.g., location: city list).
	•	Promote to grouped (e.g., add region) when any ONE occurs:
	•	Distinct values in a facet exceed ~75–100.
	•	A typical query for that facet exceeds ~500 items or creates noticeable latency.
	•	UI/mental scan lists get unwieldy (more than ~12 options).
	•	Promote to nested (e.g., region → state → city) per hot bucket when:
	•	Any group holds ~150–200+ items frequently queried together.
	•	Never rewrite history to fit new shapes: new writes carry new tokens; old records remain valid and discoverable.

⸻

7) Quality, Safety, and Governance (What It MUST Uphold)
	•	Truth preference: Warehouse results override stale memory; memory is then updated asynchronously.
	•	Auditability: Every change is traceable via claims and head version increments.
	•	Privacy & tenancy: All reads/writes are scoped to the correct org/user; no cross-leakage.
	•	Minimal exposure: Only relevant heads are used; claims/history are not injected into prompts.
	•	Graceful degradation: If any layer is unavailable, the assistant continues with remaining layers and states the limitation.

⸻

8) Observability & Signals (What It SHOULD Report)
	•	Per turn: whether warehouse was queried, rows returned, and heads consulted (names only).
	•	Per session: number of claims created, heads updated, conflicts detected.
	•	Drift/Conflicts: Count of memory vs. warehouse mismatches flagged.
	•	Latency budget: Track p95 end-to-end response time and context assembly time.

(Surfacing these as metrics/logs is out of scope here; this section defines what the system should make observable.)

⸻

9) Edge Cases to Handle (Expected Outcomes)
	•	User correction: “Actually, hours changed.” → Answer acknowledges change; future turns reflect it; memory supersedes prior head.
	•	Ambiguous entity: “Show revenue for ‘Lineage’.” → Ask clarifying question or use memory to disambiguate org/location; then proceed.
	•	Empty warehouse: “Find last week’s invoices.” → Reply that none found; offer next steps (date widen, source check).
	•	Contradiction: Memory says A, warehouse shows B → State the current authoritative B; memory flagged for update; no argument with the user.
	•	Privacy refusal: If a user asks the bot to recall data that should not be surfaced (per policy), the bot declines and explains briefly.

⸻

10) Acceptance Criteria (You can check these and move on)
	1.	Natural-language behavior: Answers are human-readable, succinct, and adapt tone to remembered preference.
	2.	Context composition: Each answer demonstrably uses (a) last turns, (b) relevant heads, and (c) warehouse results when needed.
	3.	History integrity: New knowledge creates a claim; head version increments; prior claims remain queryable.
	4.	Conflict handling: When the warehouse contradicts memory, reply uses warehouse, and a superseding memory event is created later.
	5.	Scale promotion: When a facet breaches thresholds, new writes are grouped by an added dimension; existing data remains valid.
	6.	No hard deletes: Nothing in memory is destroyed; only superseded.
	7.	Isolation: A user/org never sees another user’s/org’s memory or chat history.
	8.	Zero-regret fallback: On retrieval or query failure, the bot still replies clearly with limitations and options.

⸻

11) Glossary (Shared Language)
	•	Facet: A category of knowledge (e.g., hours, payments, people, process, tool, boundary).
	•	Scope: The subject the facet applies to (e.g., org, location:yolkrest, person:julie).
	•	Head: The current, authoritative value for a facet+scope (fast to read at runtime).
	•	Claim: An immutable event describing an extracted statement, update, or reclassification (full history).
	•	Supersede: Establish that a new claim replaces the prior one for a given head; increases head version.
	•	Refiner: The scheduled process that turns raw chat into structured memory (claims + updated heads).

⸻

12) What You Will Get Back From The System (User-visible Promise)
	•	Natural-language replies that:
	•	Remember who/what/where from previous sessions.
	•	Default to your policies, terms, and preferences without re-asking.
	•	Pull in live, current data from your warehouse when you ask for it.
	•	Adjust course mid-conversation when you change your mind or add constraints.
	•	Explain limitations clearly when data is missing or ambiguous.

⸻

Final Note

This PRD is complete for behavior, contracts, and expectations. Another agent can now decide how to realize the Firestore layouts, background refinement, and warehouse access that satisfy these requirements — without you needing to return for more detail.