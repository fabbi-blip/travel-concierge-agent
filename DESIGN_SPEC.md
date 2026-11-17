# DESIGN_SPEC.md — Smart Travel & Expense Concierge (`travel-concierge-agent`)

## 1. Overview

**Smart Travel & Expense Concierge** is a production-grade multi-agent concierge system built with the **Google Agent Development Kit (`google-adk`)** and deployed on **Google Cloud Run**. It automates end-to-end trip research, policy-compliant budget optimization, safety/advisory verification, reservation booking, and post-trip expense reimbursement while enforcing human-in-the-loop (HITL) financial controls, active PII redaction, OpenTelemetry tracing, and cross-session traveler memory.

- **Assessment Track**: Concierge Agents
- **Deployment Target**: Google Cloud Run (`FastAPI` + `Dockerfile` + `otel_to_cloud=True`)
- **CI/CD Runner**: GitHub Actions (`.github/workflows/`) + Terraform IaC (`deployment/terraform/`)

---

## 2. Architecture & Multi-Agent Orchestration

### Agent Hierarchy (`app/agent.py` & `app/sub_agents/`)
1. **`travel_concierge_agent` (Root Coordinator — `LlmAgent`, Fast Tier)**:
   - Entrypoint coordinator that greets travelers, preloads long-term memories (`PreloadMemoryTool`), handles quick airport/advisory lookups, and routes multi-step planning vs. booking/expense execution to specialized sub-agents.
2. **`itinerary_planning_pipeline` (`SequentialAgent`)**:
   - **`parallel_discovery_squad` (`ParallelAgent`)**: Executes three concurrent specialists with isolated `output_key`s:
     - `flight_research_specialist` (`output_key="discovered_flights"`)
     - `accommodation_specialist` (`output_key="discovered_hotels"`)
     - `destination_advisory_specialist` (`output_key="destination_advisories"`)
   - **`itinerary_architect_pro` (`LlmAgent`, Pro Reasoning Tier)**: Synthesizes flight, lodging, and safety findings into a cohesive, budget-optimized itinerary (`output_key="draft_itinerary"`).
   - **`quality_assurance_loop` (`LoopAgent`, `max_iterations=3`)**:
     - `policy_and_budget_auditor` (`LlmAgent`, `output_schema=ItineraryAuditVerdict`, `output_key="audit_verdict"`)
     - `itinerary_refiner` (`LlmAgent`, refines `draft_itinerary` if audit flags violations)
     - `AuditEscalationChecker` (custom `BaseAgent` emitting `EventActions(escalate=True)` once `audit_verdict.approved == True`)
3. **`booking_and_expense_executor` (`LlmAgent`, Pro Reasoning Tier)**:
   - Executes reservations and expense claims strictly behind Human-in-the-Loop confirmation hooks (`require_confirmation`).

### Strategic Model Routing (`app/routing.py`)
- **Fast Tier (`gemini-2.5-flash` / configurable `FAST_MODEL`)**: Coordinator triage, concurrent discovery specialists, itinerary refinement, and `LlmEventSummarizer` event compaction.
- **Deep-Reasoning Tier (`gemini-2.5-pro` / configurable `PRO_MODEL`)**: Complex multi-constraint itinerary synthesis, structured policy self-evaluation, and financial booking/expense execution.

---

## 3. Tools & Explicit JSON Schemas (`app/tools/` & `app/schemas.py`)

Every tool uses strict Pydantic v2 input/output schemas (`ConfigDict(extra="forbid")`), comprehensive Google-style docstrings, descriptive `verb_noun` naming, and structured guided error recovery (`status`, `error_code`, `recoverable`, `recovery_instructions`):

1. `lookup_airport_iata_code`: Resolves city or airport names to 3-letter IATA codes.
2. `search_verified_flight_options`: Queries schedules, cabin classes, baggage rules, and fares with strict IATA and ISO-8601 validation.
3. `search_curated_hotel_inventory`: Queries vetted accommodations by city, dates, nightly rate ceiling, and traveler amenities.
4. `retrieve_destination_travel_advisories`: Queries persistent vector/SQLite knowledge store (and optional `VertexAiSearchTool` datastore) for visa, safety, transit, and per-diem policies.
5. `calculate_trip_budget_breakdown`: Itemizes flights, lodging, taxes, and daily per-diem against traveler budget caps with actionable savings guidance.
6. `reserve_itinerary_booking`: High-stakes reservation tool gated by `FunctionTool(..., require_confirmation=requires_booking_confirmation)`.
7. `submit_expense_reimbursement_claim`: High-stakes financial reimbursement submission gated by `FunctionTool(..., require_confirmation=True)`.

---

## 4. Context, Memory, Safety, Observability & Infrastructure

- **Constitution System Prompts (`app/prompts.py`)**: Explicit Persona, Domain Knowledge, Operational Protocol, Tool Contracts, Safety Constraints, and `{user:traveler_preferences}` state injection.
- **History Compaction & Caching (`app/agent.py`)**: `EventsCompactionConfig` + `LlmEventSummarizer`, `ContextCacheConfig`, and `ContextFilterPlugin`.
- **Persistent Session & Knowledge State (`app/app_utils/services.py`, `app/knowledge_store.py`)**: Supports Cloud SQL PostgreSQL, persistent local SQLite (`DatabaseSessionService`), Vertex AI Sessions, persistent TF-IDF vector search over travel policies/advisories, and `VertexAiSearchTool`.
- **Async Memory Consolidation (`app/memory.py`)**: Non-blocking `asyncio.create_task` in `after_agent_callback` scrubbing PII and consolidating traveler preferences into long-term memory (`VertexAiMemoryBankService` / persistent memory store).
- **Guardrails & Policy Plugins (`app/plugins/guardrails.py`)**: Custom `TravelPolicyGuardrailPlugin(BasePlugin)` screening prompt injection, unsafe requests, and policy violations (with optional Google Cloud Model Armor API integration) + `ReflectAndRetryToolPlugin`.
- **Observability, Tracing & PII Redaction (`app/observability/`)**:
  - Structured JSON logging (`google-cloud-logging` compatible JSON formatter; zero `print()` calls).
  - `IntentOutcomeAuditPlugin(BasePlugin)` capturing paired `INTENT` (before agent/tool run) and `OUTCOME` (after run + latency + status) events.
  - OpenTelemetry `TracerProvider` + `CloudTraceSpanExporter` + custom spans (`otel_to_cloud=True`).
  - `PiiRedactionService` (`google-cloud-dlp` Sensitive Data Protection API + deterministic regex scrubbing) wired into both logging and memory pipelines before storage.
- **Secrets, IaC & Automated Evals**:
  - `SecretManagerService` (`google-cloud-secret-manager`) with zero hardcoded keys.
  - Terraform configurations in `deployment/terraform/` + GitHub Actions in `.github/workflows/`.
  - Golden ADK evaluation datasets in `tests/eval/` + comprehensive `pytest` unit tests in `tests/unit/`.
