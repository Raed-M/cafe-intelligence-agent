# وضّحها — Frontend & API Implementation Plan

## 0. New-chat handoff — read this first

This document is the source of truth for continuing the project in a new Codex chat. Do not restart product discovery or ask the user to repeat decisions already recorded here. Read this entire file and `design-system/waddehha/MASTER.md` before proposing or implementing frontend work.

### User goal

Build a premium, Arabic-first, laptop-focused interface for the existing Cafe Intelligence Agent. The desired quality is a custom high-end product—not a generic admin template. The user is learning English and may answer in English, but explanations should be translated into Arabic whenever requested or when a technical choice is difficult to understand.

### Current workspace

- Repository root: `C:\Users\hassa\OneDrive\سطح المكتب\CooP\Week-6\Final_Project`
- GitHub origin: `https://github.com/Raed-M/cafe-intelligence-agent.git`
- Branch at clone time: `master`
- Cloned commit observed: `0c13986`
- Existing Python virtual environment: `.venv`
- Existing backend code lives under `src/`, with CLI entry points under `scripts/` and scheduler under `scheduler/`.
- No Next.js or FastAPI frontend/API implementation existed when this plan was written.

### Existing backend in plain language

The backend reads weekly café data from POS, menu, traffic, staffing, inventory, supplier emails, reviews, and café profile files. It cleans the data, fans out to five specialist analysts, executes model-generated Python in restricted subprocesses, validates claims through a Critic, ranks findings, adds calendar/prayer/local-search context, creates three bilingual content ideas, generates HTML/WhatsApp reporting, pauses for human approval, saves checkpoints/memory, and supports scheduled weekly execution.

The five current specialist analysts are:

1. Sales and product mix.
2. Margin and cost.
3. Operations.
4. Voice of customer.
5. Anomaly detection.

### Backend validation already performed

- A clean `.venv` was created and dependencies from `requirements.lock` plus `pytest`/`pytest-timeout` were installed.
- Full automated suite result: **82 passed in 604.51 seconds (10:04)**.
- Preflight verified all seven configured data sources as readable/valid.
- OpenAI connectivity check succeeded for `gpt-4o`.
- Tavily connectivity check succeeded with one real search result.
- LangSmith tracing connected only when outbound sandbox/network access was allowed.
- The workflow successfully ingested/cleaned data, generated report files, checkpointed state, and paused at the human gate.
- Diagnostic reports were generated under:
  - `outputs/reports/run_193f0fc8/report.html`
  - `outputs/reports/run_a0b8cbd2/report.html`
  - `outputs/reports/run_1820656a/report.html`
- These diagnostic runs honestly returned “insufficient evidence” because the live analyst stage failed; they are not examples of successful insight generation.

### Confirmed live-backend defects

1. **Missing model validation:** `ANALYST_MODEL`, `CRITIC_MODEL`, and `CONTENT_MODEL` resolve to empty strings when unset. Preflight still reported success after API keys were present. Live analysts then received an OpenAI 400 error: `you must provide a model parameter`.
2. **Generated-code contract mismatch:** after explicitly setting all three model variables to `gpt-4o`, the model calls succeeded but all five analyst subprocesses failed. The prompt says `ANALYST_INPUTS_JSON` contains JSON; `src/tools/code_executor.py` actually sets it to the path of `run_meta.json`. Generated code attempts `json.loads(env)` and fails. Some attempts additionally requested disallowed imports such as `textblob` or `scipy.stats`.
3. **Checkpoint warning:** LangGraph warns that checkpointed `RawCafeProfile`, `SourceRegistry`, `AppSettings`, and `RuntimeCafeConfig` types will be blocked by future strict msgpack behavior.
4. **Test gap:** the 82 tests rely on injected/fake generators for important LLM paths and therefore did not catch the real prompt/executor mismatch.

Do not claim the live AI analyst stage works until these defects are fixed and a real run produces verified findings.

### Local secrets

The user stores keys outside the repository in the parent `Week-6` directory:

- `OPENAI_API.txt`
- `LANGSMITH_API.txt`
- `TAVILY_API.txt`

Never print, copy into source control, expose to the browser, or include their values in logs. Load them server-side or ask the user to configure a gitignored `.env` safely. The key values themselves are intentionally not recorded in this plan.

### Assignment constraints that must remain visible in the UI

- The system runs automatically every week; the owner should not need to open a dashboard for normal operation.
- The product is generic: adding a second café must not require application-code changes.
- Sources parse in parallel and fail independently.
- Cleaning must report rows in, dropped, repaired, and reasons.
- Analysts run in parallel and self-correct generated code in restricted subprocesses.
- Every claim is checked against computed evidence by the Critic.
- Content ideas must combine café data, local context, and date/season context.
- Output is bilingual WhatsApp-length summary plus full HTML/PDF report.
- Human approve/edit/reject is required before delivery.
- Cross-run memory, scheduling, cost/step caps, and LangSmith tracing are required.
- Demo evidence must include 10 weekly cycles, five hand-verified metrics, non-zero critic rejection behavior, fault injection, and second-café proof.
- Assignment grading weights: findings/correctness 25%, architecture 25%, grounded content 15%, testing 15%, generic second café 10%, report/demo 10%.

### Approved reference artifacts

- Complete implementation plan: this file.
- Generated design-system baseline: `design-system/waddehha/MASTER.md`.
- Approved dark logo concept reference: `design-system/waddehha/references/logo-concept-gulf-night.png`.
- The reference image is a concept only. Arabic lettering must be corrected and rebuilt manually as SVG before production use.

### Installed design guidance

The following Codex skills were installed and used during discovery:

- `frontend-design`
- `ui-ux-pro-max`

The generated UI/UX database recommendation correctly identified a data-dense dashboard, but its generic blue SaaS palette, community-landing pattern, and Latin-only Fira typography were explicitly rejected. Use the custom Gulf Night direction and Arabic-first decisions in this file instead.

### Immediate implementation starting point

When the user opens a new chat and asks to continue, the recommended order is:

1. Read this entire plan and the design-system master file.
2. Inspect current `git status`; preserve user changes and generated test evidence.
3. Fix backend defects B1 and B2 and add production-contract regression tests.
4. Run focused tests, then one real keyed weekly run and confirm non-zero verified findings.
5. Scaffold FastAPI and Next.js only after the live backend path is trustworthy.
6. Implement the smallest vertical slice: login → café → dashboard → run status → evidence → report approval.

### Pending backend update notice

The user stated that backend files have changed upstream and will explicitly request a pull with the new updates. Do **not** assume the defects or file contracts documented above still apply after that pull, and do not pull without the user's instruction. Before pulling:

1. Inspect `git status` and identify local/generated changes.
2. Preserve this plan, the design-system directory, and any user-owned work.
3. Fetch/pull non-destructively; never reset or discard local work.
4. Review the upstream diff, especially configuration, preflight, code executor, prompts, graph state, schemas, and tests.
5. Re-run preflight, focused production-path tests, the full suite where time permits, and one real keyed weekly cycle.
6. Update this handoff section with the new commit, verified behavior, remaining defects, and any changed API requirements before building the frontend.

### Information still intentionally pending

- Final names, descriptions, visual details, colors, and phrases for every agent character.
- Final manually drawn SVG logo and its complete brand package.
- Pearl Day light-theme comparison.
- Verified Saihat/Qatif dialect copy.
- Audio, music, narration samples, and approval.
- Presentation Mode implementation.
- Production WhatsApp provider credentials and social-platform publishing permissions.

## 1. Product definition

**وضحها (WADDEHHA)** is an Arabic-first café intelligence platform that turns fragmented operational data into verified findings, actions, reports, alerts, and grounded marketing recommendations.

The current reference café is **Qahwa Saihat**, but the product must remain generic. A new café is onboarded through its profile and source configuration without changing application code.

### Primary audience

- 75% technical: architecture, agent execution, evidence, critic decisions, traces, failures, cost, and data lineage.
- 25% business: revenue, margin, waste, staffing, customer experience, recommended actions, and marketing.

### Product principles

1. Lead with the weekly story, not a wall of charts.
2. Every claim must link to its calculation and source records.
3. Raw data is immutable; corrections create traceable cleaned versions.
4. Arabic is the default language and RTL direction; English is available through a global switch.
5. The owner receives proactive WhatsApp/email summaries and does not need to live in the dashboard.
6. Advanced technical detail is available through an Executive/Technical view switch.
7. Never present planned or simulated functionality as live.

## 2. Scope decision

The full product described during discovery is larger than a same-day build. Delivery is split into a functional MVP and explicit later phases.

### MVP — build first

- Fix the two blocking live-backend issues.
- FastAPI layer over the existing LangGraph workflow.
- Next.js Arabic-first application shell.
- Local email/password authentication with owner, manager, and employee roles.
- Café selector with Qahwa Saihat and the existing second-café fixture.
- Weekly overview dashboard backed by real pipeline/report data.
- Data source explorer with raw/cleaned comparison and lineage.
- Agent run page with status, errors, candidate findings, critic decisions, and evidence.
- Report viewer with manager review and owner approve/edit/reject controls.
- Bottom-center Ask Your Data composer with a narrowly grounded first implementation or an explicitly labelled demo fallback.
- Local one-command startup and a reliable seeded demo state.

### Phase 2

- Full grounded text and voice Ask Your Data.
- Demand forecasting.
- Menu engineering.
- Waste-to-riyals optimization.
- Hybrid real-time alerts and business thresholds.
- Action Center for assigning alerts/recommendations as tasks.
- Usage/cost dashboard.
- Notification settings for WhatsApp, email, and in-app delivery.
- Configurable report templates and inline report editing with version history.

### Phase 3

- Marketing Studio for TikTok and Instagram.
- Bilingual hooks, captions, formats, recommended products, and posting times.
- Generated social visuals and downloadable publishing packages.
- Marketing calendar, approval states, and campaign history.
- Competitor Watch based on public local information.
- Social platform integrations and campaign-to-sales measurement.
- External POS, email, Google Drive, and OneDrive integrations.
- Multi-branch comparison.
- Cloud deployment.

### Deferred presentation layer

The cinematic presentation is deliberately separate and deferred. It will later live at `/presentation`, share real dashboard components, and combine a short video intro with presenter-controlled interactive scenes.

## 3. Required backend repairs

These must be completed before judging the live UI.

### B1 — model configuration validation

`config/app_settings.yaml` interpolates `ANALYST_MODEL`, `CRITIC_MODEL`, and `CONTENT_MODEL` to empty strings when they are unset. Preflight currently does not reject this.

**Fix:** provide safe defaults or fail preflight with a precise message. Validate provider, key, and every required model name.

### B2 — generated-code input contract

The analyst prompt says `ANALYST_INPUTS_JSON` contains JSON, while `code_executor.py` sets it to the path of `run_meta.json`. Real generated analysts call `json.loads()` on the path and fail.

**Fix:** choose one contract and enforce it everywhere. Recommended: rename the variable to `ANALYST_INPUTS_PATH`, instruct generated code to read the referenced file, and update tests to exercise the exact production contract.

### B3 — checkpoint compatibility

LangGraph warns that unregistered Pydantic/config objects in checkpoint state will be blocked in a future version.

**Fix:** store JSON-native config summaries and re-resolve runtime configuration from stable paths/IDs, or explicitly register allowed types after a security review.

### B4 — production-path test

Add at least one integration test using the same environment-variable and generated-code contract as production. Existing injected generators pass while the real model path fails.

## 4. Technical architecture

```text
Next.js web app
    │ REST + SSE
    ▼
FastAPI API
    ├── Authentication / RBAC
    ├── Café and source configuration
    ├── Upload and data exploration
    ├── Run orchestration
    ├── Reports and approvals
    ├── Chat/query service
    └── Notifications and tasks
          │
          ▼
Existing LangGraph backend
    ├── Parsers and cleaning
    ├── Five specialist analysts
    ├── Critic and ranking
    ├── Context and content
    ├── Reporting
    └── SQLite checkpoint/memory
```

### Frontend stack

- Next.js with TypeScript and App Router.
- Tailwind CSS with CSS design tokens.
- Accessible headless components (Radix/shadcn primitives where useful, visually restyled).
- TanStack Query for server state.
- TanStack Table for large data tables.
- ECharts or Recharts for accessible business charts.
- Framer Motion for component motion; GSAP only for special orchestrated scenes.
- React Flow for the technical agent graph.
- next-intl for Arabic/English localization.
- Zod for client-side schemas.

### API stack

- FastAPI and Pydantic.
- Cookie-based secure sessions for the local web app.
- Argon2/bcrypt password hashing.
- SQLite initially, with repositories that can later move to PostgreSQL.
- Server-Sent Events for run progress; WebSockets only if bidirectional live interaction becomes necessary.
- Background execution separated from HTTP request lifetime.

## 5. Proposed repository structure

```text
Final_Project/
├── api/
│   ├── main.py
│   ├── dependencies.py
│   ├── schemas/
│   ├── routers/
│   │   ├── auth.py
│   │   ├── cafes.py
│   │   ├── sources.py
│   │   ├── runs.py
│   │   ├── reports.py
│   │   ├── approvals.py
│   │   └── chat.py
│   └── services/
│       ├── graph_service.py
│       ├── data_service.py
│       └── report_service.py
├── frontend/
│   ├── app/
│   │   ├── (auth)/login/
│   │   └── (workspace)/
│   │       ├── dashboard/
│   │       ├── alerts/
│   │       ├── data/
│   │       ├── agents/
│   │       ├── analysis/
│   │       ├── conversations/
│   │       ├── reports/
│   │       ├── marketing/
│   │       └── settings/
│   ├── components/
│   ├── features/
│   ├── lib/
│   ├── messages/ar.json
│   └── messages/en.json
├── src/                    # existing backend
├── tests/
│   ├── api/
│   └── frontend/
└── design-system/waddehha/
```

## 6. Information architecture

### Sidebar

**Overview**

- Dashboard
- Alerts
- Action Center

**Data**

- Upload data
- Data explorer
- Data quality
- Sources
- Local context

**Analysis**

- Agents
- Findings
- Forecasting
- Menu engineering
- Waste

**Communication**

- Ask your data
- Conversations
- Marketing Studio

**Control**

- Reports and approvals
- Usage and cost
- Users and roles
- Integrations
- Settings

## 7. Core page specifications

### 7.1 Dashboard

1. **Weekly story:** one clear bilingual summary of what happened, why, and what to do.
2. KPI strip: net revenue, gross profit, conversion, waste cost, and labour cost.
3. Critical alerts and opportunities.
4. Recommended actions ordered by impact and urgency.
5. Current/latest run with live agent states.
6. Three grounded content ideas.
7. Comparison with previous weeks and seasonal context.

Every card supports **Show evidence**, which opens the metric calculation, cleaned rows, and original records.

### 7.2 Data explorer

- Source, café, branch, date, column, product, and status filters.
- CSV, Excel, JSON, text-email, and normalized table views.
- Side-by-side raw/cleaned comparison.
- Changed cells highlighted with the repair reason.
- Immutable raw record and versioned corrections.
- Deep links from findings to result artifact, cleaned rows, and raw rows.
- Export only the currently authorized filtered result.

### 7.3 Agents

- Six illustrated symbolic characters: five analysts plus Critic; Content joins as a later character.
- Faces, expressions, idle/working/success/revision/failure states.
- Each character has a user-supplied name, description, role, tone, color, and signature phrase.
- Executive card explains the conclusion and impact.
- Technical drawer shows inputs, generated code, subprocess result, revision attempts, critic feedback, trace, runtime, tokens, and cost.
- Authorized users may rerun one agent; the rerun must return through Critic validation.

### 7.4 Ask your data

- Persistent centered bottom composer on workspace pages.
- Arabic RTL by default; English supported.
- Text, microphone, attachments, café/date context chips.
- Expands upward without leaving the current page.
- Full conversation-history page for rename, search, pin, share, export, archive, and delete.
- Answers must include period, sources, calculations, confidence, and limitations.
- Proposed actions require preview and confirmation.
- Voice transcript is editable before submission.

### 7.5 Reports and approvals

- Weekly executive, technical, monthly, and custom templates.
- Arabic, English, or bilingual output.
- HTML, PDF, and WhatsApp summary.
- Manager review followed by owner approve/edit/reject.
- Owner may delegate final approval to a manager.
- Inline text editing, selective paragraph regeneration, comments, comparison, and version history.
- Computed numbers cannot be manually overwritten.

### 7.6 Local context

- Map/location, weather, local events, Ramadan/Eid, prayer times, exam seasons, and search results.
- Shows exactly how context affected a finding, recommendation, or posting time.
- Labels facts, inferences, and unavailable information separately.

### 7.7 Marketing Studio

- TikTok/Instagram ideas grounded in sales, margin, stock, reviews, date, and local context.
- Arabic/English hooks and captions.
- Reel/carousel/story/trend-audio formats.
- Recommended product, day, time, and supporting finding.
- Generated visual preview and downloadable publishing pack.
- Calendar and workflow states: idea, review, approved, scheduled, published.
- Competitor Watch uses public information and clearly labels inference.

### 7.8 Alerts and Action Center

- Automatic anomaly detection plus configurable hard thresholds.
- Informational, important, and critical severity.
- Explanation, evidence, expected impact, and suggested action.
- Acknowledge, assign, snooze, resolve, or convert into task.
- Tasks include assignee, deadline, priority, comments, and optional completion evidence.

## 8. Roles and permissions

### Owner

- Full financial data and all cafés/branches.
- Users, roles, retention, integrations, model, budget, export, and deletion.
- Final report/content approval.

### Manager

- Upload/map data, run/rerun analyses, inspect reports, alerts, tasks, forecasts, and costs.
- Review and edit before owner approval.
- No access to raw API keys or billing credentials.

### Employee

- Assigned operational tasks and permitted alerts.
- Limited inventory/waste/schedule information.
- No full profit, model-cost, user-management, or destructive controls.

### Demo visitor

- Read-only seeded mock workspace.
- No access to secrets, uploads, live runs, or destructive actions.

## 9. Brand and design system

### Approved identity

- Product name: **وضحها | WADDEHHA**.
- Signature: interactive pearl representing clarity and verified insight.
- Logo direction: flowing coastal Arabic wordmark (concept B), with the dot of **ض** represented by a small pearl and a separate legible shadda.
- Final logo must be manually reconstructed as an SVG; generated images are reference material only.
- Default visual direction: **Gulf Night**. A Pearl Day variant will be explored later.

### Custom palette

The generic generated blue SaaS palette is rejected in favor of:

| Token | Purpose | Initial value |
|---|---|---|
| Gulf Night | Primary background | `#061426` |
| Deep Water | Elevated surfaces | `#0B2036` |
| Pearl | Primary text/signature | `#F6F2E9` |
| Palm | Success/local accent | `#2F6B55` |
| Copper | CTA/focus accent | `#B96F3A` |
| Sand | Warm muted surface | `#CBBEA5` |
| Coral | Critical state | `#D65C5C` |

All final token pairs must be contrast-tested to WCAG AA.

### Typography

- Arabic display/brand: custom wordmark; interface candidate must be tested (e.g. IBM Plex Sans Arabic, Alexandria, or Noto Kufi Arabic).
- Arabic body: highly legible UI face with complete weights.
- English body: complementary sans face.
- Technical/data: monospaced face used only for IDs, code, and traces.
- Minimum body size 16px; no decorative Arabic font for tables or long text.

### Motion

- One signature motion system centered on the pearl and agent states.
- Normal UI transitions: 150–300ms.
- Never animate dense table geometry.
- Respect `prefers-reduced-motion`.
- Sound is muted by default and remains deferred with Presentation Mode.

## 10. API contract (initial)

```text
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/auth/me

GET    /api/cafes
POST   /api/cafes
GET    /api/cafes/{cafe_id}

GET    /api/cafes/{cafe_id}/sources
POST   /api/cafes/{cafe_id}/sources/{source}/upload
GET    /api/cafes/{cafe_id}/data/{source}
GET    /api/cafes/{cafe_id}/data/{source}/{record_id}/lineage

POST   /api/runs
GET    /api/runs/{run_id}
GET    /api/runs/{run_id}/events
POST   /api/runs/{run_id}/agents/{agent}/rerun
GET    /api/runs/{run_id}/findings

GET    /api/runs/{run_id}/report
POST   /api/runs/{run_id}/manager-review
POST   /api/runs/{run_id}/decision

POST   /api/conversations
GET    /api/conversations
POST   /api/conversations/{id}/messages
```

All responses use stable IDs, ISO timestamps, explicit café/branch IDs, schema versions, and structured error objects. No endpoint returns filesystem paths directly to the browser.

## 11. MVP implementation order

1. Repair backend production contracts and add regression tests.
2. Create FastAPI health, café, run, event, report, and approval endpoints.
3. Scaffold Next.js, localization, tokens, layout, sidebar, café selector, and auth.
4. Build dashboard using existing saved report/artifact data.
5. Build run/agent technical inspection and SSE progress.
6. Build report approval flow.
7. Build data explorer raw/cleaned comparison.
8. Add bottom-center chat shell and a clearly scoped grounded endpoint.
9. Seed local users and deterministic demo workspace.
10. Test, capture screenshots, and prepare recovery instructions.

## 12. Validation checklist

### Functional

- Login and RBAC verified for all three roles.
- Café switching never leaks data between cafés.
- Run starts, streams status, pauses at HITL, and resumes after decision.
- Real analyst path produces findings after backend fixes.
- Report and WhatsApp summary render Arabic/English correctly.
- Raw/cleaned lineage links are reproducible.
- Failed source degrades visibly without crashing the run.

### UX/accessibility

- RTL/LTR switch changes layout direction, not only text.
- Full keyboard navigation and visible focus states.
- Touch targets at least 44×44px.
- Contrast meets WCAG AA.
- Tables do not create page-level horizontal overflow.
- Loading, empty, partial, failure, and permission-denied states give a next action.
- Reduced-motion mode works.
- Bottom composer never covers page controls or the last table rows.

### Performance

- Lazy-load heavy charts, React Flow, and future 3D pearl.
- Virtualize large tables.
- Avoid sending full raw datasets to the browser.
- Paginate/filter on the API.
- Reserve component dimensions to prevent layout shift.

## 13. Honest demo/status language

Use these labels consistently:

- **Live:** connected to and produced by the real backend.
- **Cached:** previously generated real result.
- **Simulated:** scripted interface behavior using fixture data.
- **Planned:** designed but not implemented.
- **Degraded:** run completed with missing/failed capabilities.

This distinction protects the credibility of the project with a technical audience.

## 14. Definition of done for the first release

The first release is done when a user can log in, choose a café, inspect valid source data, start a real weekly run, watch agent state changes, open a verified finding and trace it to evidence, review the bilingual report, and complete the human approval decision—all locally through the new interface.

## 15. Agreed decision register

This section records the detailed decisions made during discovery so they are not lost when implementation begins.

### Positioning and audience

- Brand name: **وضحها | WADDEHHA**.
- 75% technical audience and 25% business audience.
- The interface must prove the multi-agent architecture while translating it into business impact.
- The product is generic for different cafés; Qahwa Saihat is the current cultural and demonstration identity.
- The initial visual atmosphere draws from Saihat/Qatif agriculture, palms, sea, pearl diving, calmness, hospitality, and community belonging.
- Local dialect terminology is postponed until it can be reviewed accurately with people from Saihat/Qatif.

### Product modes

- The live dashboard is the current priority.
- A separate Presentation Mode is deferred and will not complicate the first dashboard build.
- Future Presentation Mode will combine a short cinematic video with a coded interactive demo.
- The presenter controls the sequence manually rather than relying on autoplay.
- Planned presenter tools: next/previous, pause, chapter jump, live-run jump, controlled failure, reset, timer, keyboard shortcuts, and private presenter notes.
- The presentation strategy uses cached real results for reliability plus a clearly labelled genuine live run.
- A controlled corrupt-Excel scenario demonstrates graceful degradation, followed by a fast reset.

### Language and devices

- Arabic is the default language and RTL is the default direction.
- A global switch changes the full application to English/LTR.
- Technical terms may remain English where translation reduces clarity.
- Laptop/projector is the primary experience.
- Desktop provides the full operational and technical interface.
- Tablet supports most management functions.
- Mobile focuses on summaries, alerts, chat, tasks, and approve/reject actions rather than compressed desktop tables.

### Themes and identity

- Approved primary exploration: **Gulf Night**.
- A light **Pearl Day** concept will be produced later for comparison.
- Signature application element: an interactive pearl whose state reflects ingestion, verification, failure, and completion.
- The pearl may use lightweight Three.js/WebGL with a static fallback and reduced-motion support.
- Approved logo concept: coastal flowing wordmark direction B.
- The dot of Arabic **ض** is a small pearl; the shadda remains separate and legible.
- Approved concept-board treatment: dark Gulf-night background, pearl-white lettering, palm-green wave, and restrained copper details.
- The final mark is not an AI-generated raster asset. It must be manually reconstructed, corrected, optically spaced, and delivered as SVG plus light, dark, monochrome, icon, favicon, and small-size variants.

### Agent characters

- Agents are illustrated symbolic characters—not realistic humans and not robots.
- They have faces, expressions, and interactive states.
- States include idle, working, finding discovered, revision requested, failed, and verified.
- Each personality matches its function rather than sharing one generic tone.
- Sales is energetic; Margin is careful; Operations is organized; Customer is empathetic; Anomaly is curious; Critic is calm and strict; Content is creative but evidence-bound.
- The user will later provide each character's name, description, role, appearance, language, color, and signature phrase.
- Technical names remain visible under character names.

### Dashboard and navigation

- The dashboard opens with the **Weekly Story**, followed by KPIs, alerts, actions, run status, content ideas, and historical comparison.
- Executive and Technical views coexist behind a clear switch.
- Sidebar groups: Overview, Data, Analysis, Communication, and Control.
- Sidebar is collapsible for laptop table/chart space.
- A global command menu opens with `Ctrl/Cmd + K`.
- Planned shortcuts include navigation, search, new analysis, upload, close, and shortcut help.
- Every keyboard action also remains available through visible controls.

### Ask your data

- Persistent composer is centered at the bottom of the workspace, matching the supplied Codex-style reference.
- It expands upward, retains the page behind it, and never covers controls or the final table rows.
- It understands the active café, branch, date range, page, and selected records.
- It supports Arabic/English text and push-to-talk voice.
- Voice transcript is editable before sending; spoken answers are optional and can be stopped, replayed, or muted.
- Audio is not permanently stored unless the owner explicitly enables retention.
- Answers show sources, period, calculations, confidence, and limitations.
- The assistant may prepare actions but must show a preview and request confirmation before execution.
- A separate Conversations page stores, searches, renames, pins, shares, exports, archives, and deletes prior conversations.

### Data and onboarding

- Multiple cafés/branches are supported structurally from the beginning.
- Data and memory remain separated by café/branch.
- New-café onboarding covers profile, location, hours/Ramadan hours, social handles, sources, sample upload, column mapping, quality checks, schedule, recipients, and trial run.
- Owner creates the café and may delegate source mapping/upload to a manager.
- Current ingestion uses CSV, Excel, JSON, and plain-text email uploads.
- Integrations page is designed now for later POS, email, Google Drive, and OneDrive connections.
- Raw records are immutable.
- Cleaned records link directly to their raw originals.
- Side-by-side before/after comparison highlights changed cells, reason, rule/agent, time, and run ID.
- Correcting data creates a version; it never overwrites the original.

### Authentication, permissions, and data control

- Use real email/password authentication—not simple role buttons.
- Seed local owner, manager, and employee accounts for development.
- Owner has final approval and full data/configuration control.
- Manager reviews first and may receive delegated final approval.
- Employee access is limited to assigned operational data and tasks.
- Demo visitor is read-only and uses fixture data only.
- Owner can configure retention, export café data, archive, and permanently delete after strong confirmation.
- All exports, corrections, approvals, assignments, and destructive actions enter the audit log.

### Reports and approvals

- Manager reviews and comments before sending to the owner.
- Owner approves, edits, or rejects; owner can delegate final approval.
- Report states are explicit: analyzing, manager review, owner review, approved, rejected, and delivered.
- Templates: weekly executive, technical, monthly, custom, WhatsApp, HTML, and PDF.
- Users may edit prose, regenerate one section, change chart/period, comment, and compare versions.
- Computed figures cannot be manually overwritten.

### Notifications

- Channels: WhatsApp, email, and in-app.
- Settings capture international WhatsApp number, email, verification status, preferred language, recipient role, message categories, severity threshold, and enabled/disabled state.
- Include a **Send test message** action.
- Mask phone numbers and emails from unauthorized users.
- Real WhatsApp delivery requires a future WhatsApp Business API/provider integration; collecting a number alone does not send messages.

### Added intelligence features

- Ask Your Data.
- Menu engineering.
- Waste-to-riyals analysis and order recommendations.
- Demand forecasting per item.
- Hybrid real-time alerts: statistical anomalies plus user-defined thresholds.
- Three alert levels: informational, important, and critical.
- Alerts can be acknowledged, assigned, snoozed, resolved, or converted into tasks.
- Action Center tasks contain assignee, deadline, priority, status, comments, and optional proof of completion.
- The system later measures whether completing an action improved the related metric.
- Local Context includes map/location, weather, local events, seasons, Ramadan/Eid, exams, and prayer times.

### Marketing Studio

- Business intelligence remains the product core; Marketing Studio is a major supported area.
- Supports TikTok and Instagram ideas, bilingual hooks/captions, formats, products, evidence, and recommended posting time.
- Includes generated visual previews, mobile preview, approval, downloadable publishing pack, calendar, status, and history.
- Competitor Watch uses only public information and distinguishes verified facts from inference.
- Direct social publishing and performance-to-sales measurement are later integrations.

### Models, observability, and cost

- First version uses one model configuration across the system, initially OpenAI.
- Provider abstraction must allow Claude or Gemini later.
- Do not depict GPT, Claude, and Gemini as simultaneously active when they are not.
- Future advanced settings may assign different models per agent or compare models for selected findings.
- Usage and Cost is visible to owner and manager, not employee.
- It shows per-run/per-agent tokens, cost, time, steps, weekly/monthly totals, estimates, warnings, and caps.
- Manager never sees raw API keys or billing credentials.
- Technical View links to LangSmith traces when available.

### Local deployment and audio

- Current release runs locally only.
- Next.js, FastAPI, databases, and background execution start through one documented command or Docker Compose.
- Cached/demo states remain usable without internet; OpenAI/Tavily live capabilities clearly report connectivity requirements.
- API keys remain server-side.
- Cloud deployment is a future phase.
- General dashboard audio remains off.
- Future presentation audio is muted by default and starts only after explicit user choice.
- Music, effects, and Arabic narration are separate controls.
- Every proposed sound must be previewed and approved before inclusion.

### Time and scope constraint

- Planning deadline recorded during discovery: 6 August 2026 at 11:00 PM Riyadh time.
- A polished, truthful vertical slice has priority over unfinished breadth.
- Planned features must be labelled as planned; simulations must be labelled as simulated.

## 16. Implementation status — 2026-08-06

This section records the implementation state without rewriting the historical
plan and decision register above.

### Repository and delivery state

- The repository was pulled fast-forward-only to `b02dd21`.
- Pre-pull local evidence was preserved in stash
  `codex-preserve-pre-b02dd21-test-evidence`.
- No commit or push was performed by Codex; the user retains that step.

### Backend repairs completed

- Model configuration now has a preflight check before expensive execution.
- Generated analyst results use a strict result contract and bounded repair
  path.
- Checkpoint serialization uses the reviewed `JsonPlusSerializer` allowlist.
- Critic verification now checks evidence, periods, direction, and semantic
  rejection conditions.
- Revision fan-in uses an explicit reducer rather than silently replacing
  sibling results.
- Invalid content fails closed after repair exhaustion.
- A graph paused before the human gate is represented as `waiting_review`.

### Live verification recorded

- Live run `run_6fe6dbc3` analyzed the week beginning `2026-07-20` through
  real OpenAI `gpt-4o`, Tavily, and LangSmith integrations.
- It produced 5 candidates: 4 critic-approved, 1 rejected, 4 ranked final
  findings, and 3 valid content ideas.
- The run paused before the human gate with status `waiting_review`.
- This was a degraded run: the Margin, Operations, and Anomaly agents failed;
  it must not be presented as a fully healthy all-agent run.

### Web product vertical slice

- The FastAPI MVP vertical slice is implemented: cookie authentication and
  RBAC, cafés, saved/live runs, SSE events, findings/evidence, reports,
  manager review, owner decision, lineage, and narrowly grounded chat.
- The Next.js frontend MVP vertical slice is implemented against that API
  contract.
- A Windows local launcher validates the existing `.venv` and
  `frontend/node_modules`, then starts and owns the FastAPI and Next.js child
  processes without installing packages or embedding secrets.

### Verified test and build results

- Full backend coverage passed in partitions covering the complete suite:
  88 tests across unit, graph, grounding, fault, API, and second-café coverage;
  16 integration tests; and 1 ten-week-cycle test, for 105 passing tests total.
- The monolithic backend command was externally timed out after 20 minutes and
  51 progress dots. The complete partitioned runs above were used instead of
  presenting that interrupted command as a pass or failure.
- The focused API subset passed 12 tests with 1 non-failing Starlette warning.
- Frontend lint and TypeScript type-check both exited successfully; Vitest
  passed 5 of 5 tests.
- Playwright on Edge completed 1 of 1 real-login E2E test with a clean exit. It
  verified RTL/LTR behavior, no overflow, sidebar and composer assertions, and
  exact child-process cleanup.
- The Next.js 16.3 production build completed successfully with 8 routes.
- The offline production-dependency audit reported 0 vulnerabilities.
- Verified screenshots are stored at
  `outputs/test_evidence/ui/dashboard-desktop-ar.png` and
  `outputs/test_evidence/ui/dashboard-mobile-en.png`.

### Remaining limitations and deferred scope

- Generated-code execution is best-effort isolation, not a security boundary
  for hostile code.
- The recorded live run remained degraded because three analyst branches
  failed.
- The archived item described as a 29-week scan contains 21 completed weeks;
  its stored machine paths are stale and some recorded hashes changed after
  repository normalization.
- The core owner `edit` route can regenerate a report, but requested prose
  edits are not yet applied as a complete editing workflow.
- Phase 2/3 breadth, the final logo, agent characters, and external
  integrations remain planned/deferred rather than implemented.

## 17. Continuation status — 2026-08-07

### Upstream backend refresh

- Local frontend/API work and evidence were preserved in
  `codex-preserve-webapp-before-d0f6296-backend-pull`.
- `master` was pulled fast-forward-only from `b02dd21` to `d0f6296`.
- The incoming backend adds a dedicated `cross_domain_synthesis` graph stage,
  finding hardening, key rotation, local LangSmith support, PDF rendering, and
  new live-validation evidence.
- No tracked backend file was changed after the pull. No commit or push was
  performed.

### Frontend alignment and UX verification

- The live technical workflow now displays seven stages and maps the new
  `cross_domain_synthesis` event to its own interactive node between analyst
  fan-out and Critic verification.
- Dashboard sidebar navigation now selects exactly one section, including the
  short-page/bottom-of-page Reports case, and returns to Agents as the user
  scrolls upward.
- The Ask Your Data composer keeps its hover preview, opens reliably from the
  pearl button, focuses the textarea, and stays closed after its X/Collapse
  action until the user intentionally returns.
- Vitest resolves the same `@/` alias used by Next.js, and the E2E locator for
  the chat Collapse action is exact.

### Verification results on `d0f6296`

- New-backend focused tests: **48 passed** (cross-domain, Critic, WhatsApp,
  graph revision, and fault injection).
- Frontend ESLint: passed with zero warnings.
- Frontend TypeScript: passed.
- Frontend Vitest: **6/6 passed**.
- Next.js 16.3 production build: passed with **8 routes**.
- Playwright Edge E2E: **1/1 passed**, covering signup, pending login,
  owner approval, English workspace, single-active sidebar, live workflow,
  Ask Your Data open/send/close, and file selection at desktop/mobile widths.
- Existing FastAPI adapter suite: **14 passed, 1 failed**. The remaining test
  expects a `pos` item from the newest included evidence checkpoint, but the
  selected upstream checkpoint does not contain that source. It was not fixed
  because this continuation explicitly forbids backend changes.
- No new keyed weekly network run was started during this frontend-only
  continuation; the prior verified/degraded live-run record remains the latest
  locally documented genuine run.

### Local access state

- The local development owner is now `admin` / `admin` and is the only login
  account. Three legacy development users were removed.
- A recoverable pre-migration database copy is stored at
  `outputs/test_evidence/api-before-admin-purge-20260807.sqlite3`.

## 18. Owner-managed AI connections — 2026-08-07

- Added an English, owner-only `AI connections` workspace page and sidebar
  destination. It labels the selected provider key and analysis model as
  required, and Tavily, LangSmith, and a separate utility model as optional.
- Added model cards for OpenAI, Anthropic, and Gemini with plain-language
  comparisons. The initial three-bar approximation was removed after review.
  Cards now show exact standard paid input/output USD prices per million text
  tokens, context size, a written comparative speed tier, lifecycle status,
  pricing caveats, and the best-fit workload. Provider-native default analysis
  and utility models are returned by the API catalog.
- OpenAI includes GPT-5.6 Sol/Terra/Luna plus GPT-4o mini and GPT-4o as
  clearly marked previous-generation compatibility choices. Anthropic now
  includes Fable 5, Opus 5, Sonnet 5, and Haiku 4.5. Gemini now includes
  Gemini 3.1 Pro Preview, Gemini 3.6 Flash, and Gemini 3.5 Flash-Lite.
- Each provider comparison links to its official current model/pricing page.
  The page explicitly distinguishes token price from token quantity and says
  that speed labels are comparative guidance rather than a latency guarantee.
- Added owner-only API routes to inspect, save, test, and remove runtime AI
  settings. Responses expose only configured flags and short SHA-256
  fingerprints; they never return secret values and use `Cache-Control:
  no-store`.
- Session-only configuration remains the default. The optional Remember
  control encrypts the settings with Windows DPAPI and binds decryption to the
  current Windows account. The encrypted file is `db/ai_settings.dpapi` and is
  not used when Remember is off.
- Saving applies provider and per-agent model environment variables inside the
  API process so subsequent pipeline runs use the chosen configuration without
  modifying the pulled core backend.
- The connection test sends only a minimal `OK` prompt and no café data.
- Verification: AI settings API tests **3/3 passed**, including ciphertext and
  restore checks under the real Windows user profile; ESLint and TypeScript
  passed; Vitest **6/6 passed**; Next.js production build passed with **9
  routes**; Playwright Edge E2E **1/1 passed** and covers the new page, secret
  input type, defaults, active navigation, scroll reset, and horizontal
  overflow.
- The refreshed Playwright assertions also verify GPT-4o mini's $0.15 input /
  $0.60 output prices, removal of all scale-bar UI, current Claude and Gemini
  entries, and clean 375px mobile layout after the sidebar transition settles.
- The complete adapter suite now reports **17 passed, 1 failed**. The single
  unchanged upstream-evidence failure is the previously documented missing
  `pos` item; it is outside this frontend/API-adapter addition and the core
  backend remains untouched as requested.
- No tracked core-backend file was changed, and no commit or push was made.
