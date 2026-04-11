# GEX Hosted Service Implementation Plan

## Purpose

This document proposes a production-oriented implementation plan to evolve the current local Dash-based GEX monitor into a hosted web service available at `gex.my-domain.com`, with multi-user access gated by the root-domain membership system.

The goal of this document is review and alignment before engineering execution. It is intentionally detailed so product, engineering, and operations can evaluate architecture, scope, sequencing, and risks.

---

## 1. Executive Summary

### Current State

The current repository is a local Python application with these characteristics:

- Dash UI rendered from the Python process
- Direct connection from app workers to IB Gateway / TWS
- Local file-based storage using Parquet
- No standalone user account system inside this app
- No production web deployment model

This is suitable for a single-user local workflow, but it is not yet the right architecture for a hosted membership-gated service.

### Target State

We will transform the system into a web platform with:

- membership-gated user access
- a server-side market data ingestion layer
- centralized GEX computation and storage
- a web API for current and historical data
- a browser frontend at `gex.my-domain.com`
- production deployment on DigitalOcean

### Confirmed Stack Decision

The team has already confirmed the target stack:

- Frontend: Next.js
- Frontend styling: Tailwind CSS
- Backend API: FastAPI

This plan should therefore be treated as a Next.js + FastAPI implementation plan rather than a framework selection document.

### Core Architectural Principle

Live market data and GEX should be collected centrally by backend services first, then distributed to authorized users through APIs and realtime channels. Users should not connect directly to IB Gateway from their browser sessions.

### Identity Boundary

`gex.my-domain.com` should not own signup, login, membership, or billing. Those remain on the root-domain membership site. The GEX service should only verify the root-domain session or token and use that result to allow or deny access.

### Repository Constraint

For now, the web implementation must be developed inside this repository, but it must remain isolated from the current desktop tool. The immediate goal is parallel evolution, not invasive refactoring.

That means:

- the new web application should live in a dedicated folder such as `web/`
- the existing desktop-oriented Python tool should keep working without structural disruption
- shared logic should only be extracted deliberately when there is a clear reuse boundary
- a later separation into its own repository, or a later merge into the main site repository, should remain possible

---

## 2. Product Goals

### Primary Goals

- Host the service publicly at `gex.my-domain.com`
- Require a valid root-domain membership session before accessing protected features
- Support multiple concurrent users
- Deliver current GEX and historical playback through a browser UI
- Persist GEX-specific user data such as annotations, alerts, and preferences

### Secondary Goals

- Establish a clean production deployment workflow
- Reduce operational dependence on a desktop-only local runtime
- Make the codebase easier to extend into a SaaS product

### Non-Goals for Phase 1

- Billing and subscriptions
- Mobile app
- Advanced tenant administration
- full broker execution or order management

---

## 3. Why the Current App Cannot Be Deployed As-Is

The current codebase is a strong prototype, but there are structural issues for a hosted service:

- Dash is embedded directly into the application runtime, making it harder to separate frontend, API, and worker responsibilities.
- The app assumes direct connectivity to IB Gateway/TWS, which is not a suitable user-facing architecture for multi-user hosting.
- Data is stored in local Parquet files, which is not ideal for concurrent access, account-scoped queries, or production reliability.
- The app does not yet integrate with the root-domain membership/session system.
- There is no deployment topology for reverse proxy, TLS, background workers, managed storage, or observability.

---

## 4. Recommended Target Architecture

### 4.1 High-Level Components

We recommend the following service decomposition:

1. **Web Frontend**
   - Browser-based UI for dashboards, playback, and annotations
   - Hosted at `gex.my-domain.com`

2. **API Service**
   - Protected HTTP API for current GEX, historical data, user settings, and segments
   - Realtime endpoint support via WebSocket or Server-Sent Events

3. **Market Data Ingestion Worker**
   - Connects to IB or another data source
   - Normalizes incoming market data
   - Computes GEX and derived features
   - Writes outputs to database/cache

4. **Database**
   - Persistent store for identity mapping, snapshots, strikes, annotations, alerts, and audit data

5. **Realtime Distribution Layer**
   - Delivers near-live updates to connected browser clients

6. **Reverse Proxy / Edge Layer**
   - TLS termination
   - routing to frontend and API
   - domain binding for `gex.my-domain.com`

### 4.2 Recommended Technology Stack

#### Confirmed Stack

- Frontend: Next.js
- Frontend styling: Tailwind CSS
- API: FastAPI
- Database: Managed PostgreSQL
- Realtime: WebSocket or SSE
- Background jobs: Python worker service
- Deployment: DigitalOcean App Platform or Droplet-based deployment
- Access control: delegated membership/session verification from the root domain

### Frontend Implementation Direction

The frontend should be implemented in Next.js with Tailwind CSS as the primary styling system.

This implies:

- route-level gating for protected pages
- reusable UI components instead of Dash-style page composition
- centralized design tokens through Tailwind config and app theme variables
- browser-native chart and replay controls designed for product UX rather than local tooling UX

### Repository Organization Guidance

To avoid polluting the current desktop codebase, the recommended near-term structure is:

```text
gex/
├── src/                      # existing desktop/local Python app
├── web/                      # new Next.js app and web-track docs
├── services/
│   └── worker/               # extraction target for ingestion logic
└── config/
```

In this phase:

- `src/gex_monitor/...` remains the desktop tool path
- `web/` contains all Next.js and Tailwind assets
- backend/API work can either begin in `services/api/` or remain planned until Phase 1 starts
- cross-cutting refactors should be minimized until the hosted architecture is validated

---

## 5. Data Flow Model

### Proposed Flow

1. Worker connects to upstream market data source
2. Worker fetches spot, option chain, OI, greeks, and expiry data
3. Worker computes GEX, walls, flips, regime features, and other derived analytics
4. Worker stores current snapshots and history in database
5. API reads the prepared data and serves browser clients
6. Realtime layer pushes updates to authorized users
7. User-specific actions such as annotations and alerts are stored separately and merged in the UI

### Shared vs User-Specific Data

#### Shared Market Data

- spot price
- total GEX
- gamma flip
- call wall / put wall
- strike-level GEX distributions
- time series history
- replay snapshots

#### User-Specific Data

- root membership identity mapping and access context
- saved layouts
- watchlists
- alerts
- manual regime annotations
- private notes

This separation is important for both performance and product design.

---

## 6. Critical Constraint: Market Data Licensing and Redistribution

This is the most important non-code issue in the entire plan.

If the system ingests data from IB and redistributes live or near-live derived market information to multiple users, the team must verify:

- Interactive Brokers usage terms
- exchange data licensing rules
- whether derived analytics are considered redistributable
- whether delayed vs realtime delivery changes legal obligations
- whether user count or commercial access changes compliance requirements

### Recommendation

Before building public multi-user distribution on top of IB data, assign a short legal/commercial review task to confirm whether this product model is permitted.

This review should happen before committing to launch architecture.

---

## 7. Recommended Phase Plan

### Cross-Subdomain Session Assumption

This plan assumes the root membership site can provide one of the following to `gex.my-domain.com`:

- a parent-domain cookie scoped to `.my-domain.com`, or
- a signed token or session introspection mechanism that the GEX service can verify server-side

The exact mechanism must be confirmed before implementation starts because cookie scope, `SameSite`, signature verification, expiry, refresh, and logout invalidation will directly affect the final design.

### Phase 0: Architecture and Compliance Review

#### Objectives

- align on target architecture
- confirm the product scope for v1
- review market-data licensing assumptions
- confirm the root-domain membership integration model

#### Deliverables

- approved architecture diagram
- approved v1 feature scope
- data-source compliance decision memo
- approved cross-subdomain session design

#### Exit Criteria

- team agrees on whether IB can be used for hosted redistribution
- team agrees on how `gex.my-domain.com` verifies root-domain membership

---

### Phase 1: Backend Foundation

#### Objectives

- create a production-friendly backend skeleton
- introduce persistent database
- define access-control model and API boundaries

#### Tasks

- create FastAPI service
- define database schema
- implement root-domain membership/session verification
- add API endpoints for health, current snapshots, history, and user profile
- define service config model for development, staging, and production

#### Suggested Core Tables

- `users` or `member_identities`
- `membership_context` or external session mapping
- `symbols`
- `gex_snapshots`
- `gex_strikes`
- `ohlc_bars`
- `segments`
- `alerts`
- `user_preferences`

#### Exit Criteria

- a root-domain member can access the app through verified session context
- protected requests succeed only with valid root-domain membership
- backend can read and write data from PostgreSQL

---

### Phase 2: Ingestion Worker Extraction

#### Objectives

- separate market-data ingestion from the user-facing web layer
- centralize GEX computation

#### Tasks

- refactor existing IB worker code into a standalone worker service
- move calculation logic into reusable service modules
- write computed outputs to database instead of local Parquet only
- optionally keep Parquet export as an internal backup or archive path
- implement graceful reconnect and worker health monitoring

#### Notes

Most of the current logic in:

- `src/gex_monitor/ib_client.py`
- `src/gex_monitor/gex_calc.py`
- `src/gex_monitor/features.py`
- `src/gex_monitor/time_utils.py`

can be preserved and reorganized rather than rewritten from scratch.

#### Exit Criteria

- worker can run independently
- current market data is persisted centrally
- frontend/API do not require direct IB access

---

### Phase 3: Web Frontend

#### Objectives

- replace or supersede the local Dash UI with a browser-native membership-gated web application

#### Tasks

- implement gated access flow
- implement dashboard shell
- implement current GEX view
- implement historical replay
- implement user annotations and saved preferences
- implement realtime update subscription

#### UX Guidance

The frontend should not be a thin clone of the current Dash layout. It should be designed as a browser product with:

- responsive layout
- membership-aware navigation
- clear loading and stale-data states
- protected routes
- user settings and alert management

#### Exit Criteria

- valid root-domain members can access dashboard from browser
- current and historical data render from API
- no browser session depends on direct broker connectivity

---

### Phase 4: Production Deployment on DigitalOcean

#### Objectives

- deploy the hosted service under `gex.my-domain.com`

#### Deployment Options

##### Option A: DigitalOcean App Platform

Pros:

- easier deployment and managed routing
- simpler TLS and domain setup
- easier service separation for app, API, worker

Cons:

- less control for unusual networking or broker-side requirements

##### Option B: DigitalOcean Droplet

Pros:

- full control over runtime and networking
- easier to customize reverse proxy and worker processes

Cons:

- more operational burden

#### Recommendation

If data-source connectivity allows it, App Platform is cleaner for frontend/API workloads. If broker integration requires more custom networking or host-level control, Droplet may be necessary.

#### Tasks

- provision production environment
- configure secrets and environment variables
- configure managed database
- configure domain and TLS
- set up deployment pipeline
- add health checks and restart policies

#### Exit Criteria

- site is reachable at `https://gex.my-domain.com`
- membership-gated access works
- backend and worker are healthy
- production monitoring is in place

---

## 8. Proposed API Surface

Initial API candidates:

- `GET /session`
- `GET /me`
- `GET /health`
- `GET /symbols`
- `GET /snapshots/current`
- `GET /snapshots/{symbol}/history`
- `GET /strikes/{symbol}/latest`
- `GET /replay/{symbol}`
- `GET /segments`
- `POST /segments`
- `PUT /segments/{id}`
- `DELETE /segments/{id}`
- `GET /alerts`
- `POST /alerts`

This should be versioned as `/api/v1/...` in production.

---

## 9. Membership Integration and Authorization

### Requirements

- users without a valid root-domain membership session cannot access protected data pages
- each user must be mappable to a stable membership identity
- user-specific objects must be permission-checked
- session lifecycle must be secure

### Recommended Initial Access Features

- root-domain session verification middleware
- secure cross-subdomain cookie or signed token validation
- route protection
- graceful redirect or deny flow when membership is absent or expired

### Future Access Extensions

- role-based access
- organization/team accounts
- membership tier gating

### Integration Notes

The preferred implementation is:

1. the root membership site authenticates the user
2. the root membership site issues a session artifact usable by subdomains
3. `gex.my-domain.com` verifies that artifact server-side
4. the GEX service creates or updates a local identity mapping record
5. all protected API and realtime requests enforce that verified context

If the root site only exposes a browser cookie and no verification endpoint, the team must define:

- cookie domain
- cookie signing or encryption scheme
- expiry and refresh behavior
- logout invalidation behavior across subdomains
- server-side verification contract

---

## 10. Database and Storage Strategy

### Recommended Production Storage

- PostgreSQL for application data and historical snapshots
- Redis optional for caching and realtime fanout
- object storage optional for archives and exports

### Migration Guidance

Current Parquet storage can remain temporarily for:

- archive export
- debugging
- offline analysis

But production reads for membership-gated user requests should move to database-backed access.

---

## 11. Realtime Delivery Strategy

### Recommended Options

- WebSocket if the UI needs richer bidirectional behavior
- Server-Sent Events if updates are mostly one-way from server to browser

### Initial Recommendation

Start with SSE if the UI only needs live broadcast updates. It is operationally simpler. Move to WebSocket if interactive features become more complex.

---

## 12. Security Requirements

Minimum baseline:

- HTTPS only
- secure secret management
- secure cross-subdomain session handling
- CSRF protection where applicable
- rate limiting on session validation endpoints
- audit logging for key account actions
- separation between public and internal service endpoints

---

## 13. Observability and Operations

### Required Production Signals

- API health
- worker health
- database health
- ingestion lag
- last successful market update per symbol
- membership verification errors
- page/API latency

### Suggested Tooling

- structured logs
- uptime checks
- metrics dashboard
- error tracking

---

## 14. Testing Strategy

### Unit Tests

- GEX calculations
- feature derivation
- config parsing
- membership verification helpers
- permission checks

### Integration Tests

- API endpoints
- database writes/reads
- worker-to-database flow
- realtime subscriptions

### End-to-End Tests

- membership-gated entry flow
- dashboard rendering
- historical replay
- segment annotation flow

---

## 15. Suggested Repository Evolution

### Proposed Layout

```text
gex/
├── apps/
│   ├── web/                  # Next.js frontend
│   └── api/                  # FastAPI service
├── services/
│   └── worker/               # market data ingestion and GEX computation
├── packages/
│   └── core/                 # shared domain logic / calculation modules
├── config/
├── web/
│   └── docs/
└── infra/
```

### Near-Term Working Layout

Because the current requirement is to keep the hosted web work contained while the desktop tool evolves in parallel, the recommended immediate layout is:

```text
gex/
├── src/
│   └── gex_monitor/          # current desktop/local app
├── web/                      # Next.js + Tailwind app
│   └── docs/                 # web-track planning and design docs
├── services/
│   ├── api/                  # FastAPI service
│   └── worker/               # ingestion and computation
├── config/
└── infra/
```

This gives the team:

- isolation for new web work
- minimal churn in the current desktop code
- a clean migration path if `web/` is later moved to another repository
- a clean integration path if `web/` is later merged into the main site repository

### Longer-Term Alternative

If the team later wants a more formal monorepo split:

```text
gex/
├── apps/
│   ├── web/
│   └── api/
├── services/
│   └── worker/
├── packages/
│   └── core/
├── config/
├── web/
│   └── docs/
└── infra/
```

Either structure is acceptable, but the near-term working layout is the better fit for the current constraint.

---

## 16. Risks

### Technical Risks

- IB connectivity in cloud environments may be operationally awkward
- live option chain ingestion may be expensive or slow at scale
- historical strike-level storage may grow rapidly
- realtime fanout can become expensive if payloads are not normalized

### Product Risks

- unclear scope between analytics platform and trading platform
- feature expansion before architecture is stabilized

### Compliance Risks

- redistribution rights for market-derived data
- exchange licensing obligations

### Operational Risks

- worker failures during market hours
- stale data shown to users without clear UI indication

---

## 17. Recommended Immediate Next Steps

1. Review and approve this plan internally.
2. Make a data-source/compliance decision for hosted redistribution.
3. Decide whether the initial realtime transport should be SSE or WebSocket.
4. Approve a Phase 1 technical design for membership verification, API, and database schema.
5. Approve the initial Next.js frontend shell and Tailwind design system direction.
6. Start implementation only after architecture and compliance alignment.

---

## 18. Decision Recommendation

### Recommended Path

For a real hosted product, the recommended path is:

- central ingestion worker
- centralized database
- membership-gated web frontend
- API-driven delivery
- DigitalOcean-hosted deployment

Under the confirmed stack decision, that means:

- Next.js frontend
- Tailwind CSS styling system
- FastAPI backend
- Python worker service for ingestion and computation

This is the correct architecture for multi-user access and future commercialization.

The current Dash app should be treated as a prototype and a useful source of reusable domain logic, not as the final production application.

---
