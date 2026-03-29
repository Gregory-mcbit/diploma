# README.md

# Multi-Agent AI System for Investment Portfolio Formation

## 1. Project overview

This project is a **multi-agent decision-support system for investment portfolio formation and monitoring** built around **LangGraph**, **LLM agents**, **ML models**, **RAG layers**, and external **tools** for market, news, macro, optimization, and risk analytics.

The system is designed **not** as a naive “chatbot that recommends stocks,” and **not** as an LLM that directly invents asset weights. Instead, it is a structured architecture where:

- **LLM agents** handle orchestration, reasoning, validation, critique, and explanation.
- **ML models** produce quantitative signals and scores.
- **Tools** fetch fresh data and perform numerical computation.
- **RAG layers** provide rules, cached context, and memory of prior decisions.
- **LangGraph** manages state, routing, retries, loops, and agent coordination.

The result is a system that can:

- formalize investor requirements,
- collect and normalize market context,
- rank assets,
- detect market regime,
- build a constrained portfolio,
- validate it through risk analysis,
- challenge it through a critic loop,
- explain the final decision,
- monitor the portfolio over time,
- trigger rebalancing when needed.

---

## 2. Core architectural principles

This section defines the non-negotiable architectural decisions that the coding agent must follow.

### 2.1 LLM is not the numerical engine

The LLM must **not** directly calculate portfolio weights, covariance matrices, risk metrics, optimizer outputs, or historical returns.

These tasks must be delegated to:

- Python services,
- numerical libraries,
- optimization modules,
- dedicated ML models,
- market/risk tools.

The LLM’s role is:

- to choose what to run,
- to interpret outputs,
- to coordinate agents,
- to validate logic,
- to explain the result,
- to detect contradictions.

### 2.2 The system is a decision-support platform, not a prediction oracle

The system is not framed as “predict the market perfectly.”

Its purpose is:

- to build an investment portfolio aligned with investor profile and constraints,
- to combine multiple evidence layers,
- to validate portfolio quality and risk,
- to explain the recommendation,
- to support ongoing monitoring and rebalancing.

### 2.3 The architecture is multi-agent by design

The system is intentionally decomposed into specialized agents because:

- different steps require different reasoning modes,
- modularity improves debugging,
- each agent can have different tools and prompts,
- the critic loop requires separation of generation and validation,
- this decomposition is easier to justify academically and technically.

### 2.4 Portfolio construction is hybrid

The system uses a hybrid logic:

- **multi-factor scoring** to evaluate assets,
- **regime adaptation** to adjust behavior to market conditions,
- **constrained portfolio optimization** or constrained allocation to assign weights,
- **critic validation** to reject weak or inconsistent outputs.

This means the project is neither purely quant-only nor purely LLM-driven. It is intentionally mixed.

---

## 3. Approved strategic logic of the system

The approved investment logic is:

> **multi-factor scoring + regime adaptation + constrained portfolio optimization + critic validation**

This is the backbone of the whole project.

### 3.1 Multi-factor scoring

Assets are evaluated using quantitative and textual signals. Candidate factors may include:

- momentum,
- quality,
- low volatility,
- drawdown profile,
- beta,
- rolling return behavior,
- correlation structure,
- valuation metrics,
- sentiment-derived signal,
- macro-aligned signal.

### 3.2 Regime adaptation

The system must detect the current market regime and adapt scoring or portfolio construction accordingly.

Example regime classes:

- bullish,
- bearish,
- sideways,
- high-volatility,
- risk-on,
- risk-off.

### 3.3 Constrained portfolio optimization

Portfolio formation must be performed under explicit constraints, such as:

- maximum weight per asset,
- maximum weight per sector,
- asset class restrictions,
- risk profile alignment,
- diversification constraints,
- liquidity constraints.

### 3.4 Critic validation

Any proposed portfolio must be passed through a critic loop before being accepted.

The critic must be able to:

- approve,
- request weight revision,
- request replacement of assets,
- request risk reduction,
- return insufficient confidence.

---

## 4. High-level system pipeline

The approved core pipeline is:

1. **Profile Agent**
2. **Data Agent**
3. **Feature / Scoring Agent**
4. **Regime Agent**
5. **Portfolio Construction Agent**
6. **Risk Agent**
7. **Critic Agent**
8. **Explainability Agent**

Plus a separate long-running / scheduled branch:

9. **Monitoring / Rebalancing Agent**

This sequence is the canonical reference pipeline and should be reflected in code structure, folder structure, prompts, schemas, and graph orchestration.

---

## 5. Detailed description of every agent

## 5.1 Profile Agent

### Purpose
Convert raw user input into a formal investment profile that downstream agents can use reliably.

### Inputs

- investment amount,
- investment horizon,
- target objective,
- risk tolerance,
- drawdown tolerance,
- asset restrictions,
- sector restrictions,
- country restrictions,
- rebalancing preferences,
- optional preferences such as income vs growth.

### Responsibilities

- validate completeness of profile inputs,
- normalize user language into structured fields,
- infer missing reasonable defaults if business logic allows,
- formalize constraints,
- define the target portfolio policy.

### Outputs
A structured object such as:

```json
{
  "risk_profile": "moderate",
  "horizon": "long_term",
  "target": "capital_growth",
  "constraints": {
    "max_asset_weight": 0.15,
    "max_sector_weight": 0.30,
    "allowed_asset_classes": ["stocks", "etf", "bonds"],
    "forbidden_assets": [],
    "max_drawdown_tolerance": 0.18
  },
  "rebalancing_policy": {
    "mode": "threshold_and_periodic",
    "period_days": 30,
    "drift_threshold": 0.05
  }
}
```

### Why it exists as a separate agent
Without a profile normalization layer, the whole system becomes inconsistent. Every later decision depends on a clear and structured policy.

---

## 5.2 Data Agent

### Purpose
Collect, normalize, merge, cache, and package all external data needed by the system.

### Inputs

- investor profile,
- candidate universe definition,
- current timestamp / evaluation date,
- asset identifiers,
- freshness requirements.

### Responsibilities

- fetch price history,
- fetch benchmark data,
- fetch fundamental data,
- fetch macro data,
- fetch recent news and research context if needed,
- normalize schemas,
- align time windows,
- clean missing values where possible,
- attach metadata and freshness timestamps,
- write or update News/Research RAG cache after retrieval and summarization.

### Important design note
The Data Agent must not run random expensive searches by default. For text/news context it should:

1. first consult **News/Research RAG** as cache,
2. evaluate whether the cached context is sufficient and fresh,
3. only then call external tools if data is stale, incomplete, or missing,
4. process the new results,
5. store the processed context back into the News/Research RAG.

### Outputs
A normalized data package, for example:

```json
{
  "price_data": {...},
  "fundamentals": {...},
  "macro_data": {...},
  "news_context": {...},
  "freshness": {
    "market_data_ts": "...",
    "macro_data_ts": "...",
    "news_context_ts": "..."
  }
}
```

### Why it exists as a separate agent
Centralizing data access prevents duplicated tool calls, inconsistent schemas, and uncontrolled API usage.

---

## 5.3 Feature / Scoring Agent

### Purpose
Transform raw data into structured factor values and an overall attractiveness score per asset.

### Inputs

- normalized market data,
- fundamentals,
- macro signals,
- news/sentiment context,
- investor profile if score calibration depends on risk profile.

### Responsibilities

- compute factor features,
- compute or call ML-based ranking/scoring model,
- combine quantitative and contextual features,
- estimate score confidence,
- output ranked assets.

### Example factor groups

#### Market structure factors
- 1m / 3m / 6m momentum,
- trend strength,
- moving average distance,
- rolling volatility,
- rolling drawdown,
- beta,
- downside volatility.

#### Fundamental factors
- valuation,
- profitability,
- leverage,
- quality,
- earnings stability.

#### Contextual factors
- news sentiment,
- macro alignment,
- sector-level tailwind/headwind,
- event risk indicator.

### Output
A ranked asset table such as:

```json
[
  {
    "asset": "AAPL",
    "factors": {
      "momentum_3m": 0.72,
      "volatility": 0.34,
      "quality": 0.81,
      "sentiment": 0.55
    },
    "score": 0.77,
    "confidence": 0.69
  }
]
```

### Why it exists as a separate agent
Asset evaluation and portfolio construction are different tasks. First measure asset attractiveness; only then decide portfolio composition.

---

## 5.4 Regime Agent

### Purpose
Determine the current market regime and provide regime-aware context for downstream portfolio logic.

### Inputs

- price trends,
- volatility regime indicators,
- benchmark context,
- macro indicators,
- news/research cache or fresh research if required.

### Responsibilities

- classify the current regime,
- estimate confidence in regime assignment,
- provide regime rationale,
- signal any risk-off or unstable conditions.

### Possible methods

- rule-based regime classification,
- clustering on macro/market features,
- simple ML classifier,
- hybrid rule + ML.

### Output example

```json
{
  "regime": "risk_off_high_volatility",
  "confidence": 0.78,
  "drivers": [
    "elevated volatility",
    "weak benchmark momentum",
    "negative macro tone"
  ]
}
```

### Why it exists as a separate agent
Regime awareness is too important to hide inside another agent. It affects scoring interpretation, portfolio aggressiveness, and critic expectations.

---

## 5.5 Portfolio Construction Agent

### Purpose
Generate a candidate portfolio from ranked assets under the investor policy and current regime.

### Inputs

- investor profile,
- asset ranking,
- score confidence,
- market regime,
- portfolio constraints,
- optional Knowledge RAG rules,
- optional optimization tool outputs.

### Responsibilities

- select candidate assets,
- decide which assets should be excluded despite good raw score,
- call optimizer or allocation logic,
- produce proposed weights,
- align the portfolio with regime and risk profile,
- attach rationale for inclusion/exclusion.

### Important architectural rule
The Portfolio Construction Agent may reason, but **must not manually invent numerical optimization results**. It must call deterministic tools or numerical modules for:

- constrained optimization,
- risk-budgeting,
- risk parity,
- covariance-based allocation,
- turnover-aware reweighting.

### Output example

```json
{
  "selected_assets": ["AAPL", "MSFT", "IEF", "GLD"],
  "weights": {
    "AAPL": 0.18,
    "MSFT": 0.17,
    "IEF": 0.35,
    "GLD": 0.30
  },
  "construction_notes": [
    "reduced equity exposure due to risk-off regime",
    "capped single-asset concentration",
    "added defensive allocation"
  ]
}
```

### Why it exists as a separate agent
This is the central execution step that turns analysis into an actionable candidate portfolio.

---

## 5.6 Risk Agent

### Purpose
Perform independent portfolio risk validation.

### Inputs

- candidate portfolio,
- price history,
- covariance structure,
- macro regime,
- investor profile,
- Knowledge RAG risk rules.

### Responsibilities

- compute portfolio volatility,
- compute concentration risk,
- compute sector concentration,
- inspect hidden correlation concentration,
- estimate drawdown risk,
- compute VaR / CVaR if implemented,
- validate profile alignment,
- detect policy violations,
- return an explicit risk report.

### Output example

```json
{
  "risk_metrics": {
    "portfolio_volatility": 0.16,
    "var_95": 0.08,
    "cvar_95": 0.11,
    "max_sector_weight": 0.28,
    "max_asset_weight": 0.18,
    "avg_pairwise_correlation": 0.42
  },
  "violations": [],
  "warnings": [
    "elevated duration exposure",
    "equity sleeve concentrated in large-cap tech"
  ],
  "fit_to_profile": "acceptable"
}
```

### Why it exists as a separate agent
Risk must be independently evaluated. Hiding risk logic inside the construction step weakens the design and makes validation less trustworthy.

---

## 5.7 Critic Agent

### Purpose
Act as the final reasoning-level quality gate before the portfolio is approved.

### Inputs

- investor profile,
- asset scores,
- regime report,
- candidate portfolio,
- risk report,
- Knowledge RAG rules,
- Decision Memory RAG cases,
- optional news/research context if contradiction analysis needs it.

### Responsibilities

- check logical consistency,
- verify that final weights make sense relative to scores,
- inspect whether the regime was respected,
- inspect whether risk and rationale are aligned,
- compare with prior failure/success patterns from Decision Memory RAG,
- challenge unexplained concentrations,
- reject decisions with low coherence,
- return an explicit verdict and next action.

### Possible verdicts

- `approve`
- `revise_weights`
- `replace_assets`
- `reduce_risk`
- `insufficient_confidence`

### Output example

```json
{
  "verdict": "revise_weights",
  "issues": [
    "weight concentration too high relative to moderate risk profile",
    "regime indicates caution but cyclical exposure remains elevated"
  ],
  "recommended_action": "lower cyclical equity allocation and increase defensive exposure"
}
```

### Why it exists as a separate agent
Without a critic, the graph is only a sequential pipeline. With a critic, the system becomes reflective and self-correcting.

---

## 5.8 Explainability Agent

### Purpose
Produce the final human-readable explanation of the approved recommendation.

### Inputs

- investor profile,
- final portfolio,
- selected factors,
- regime result,
- risk report,
- critic rationale,
- Knowledge RAG context,
- Decision Memory RAG for historical comparison if useful.

### Responsibilities

- explain why each major asset or sleeve is included,
- explain why certain assets were excluded,
- explain why weights look the way they do,
- summarize regime and risk context,
- state limitations and caveats,
- expose uncertainty if confidence is limited.

### Output
A structured narrative suitable for UI and audit logs.

### Why it exists as a separate agent
Explanation should be generated only after the portfolio has survived validation. It must not be mixed with construction.

---

## 5.9 Monitoring / Rebalancing Agent

### Purpose
Monitor portfolio drift and decide whether rebalancing or reassessment is required.

### Inputs

- active portfolio,
- current market state,
- updated asset scores,
- updated regime,
- latest risk metrics,
- historical decision memory.

### Responsibilities

- detect weight drift,
- detect signal decay,
- detect regime shift,
- detect risk escalation,
- propose or trigger rebalancing,
- store monitoring decisions into Decision Memory RAG.

### Trigger conditions

- periodic schedule,
- threshold breach,
- large regime change,
- sudden risk deterioration,
- large deviation from target allocation.

### Why it exists as a separate agent
Portfolio formation is not a one-off act. A real portfolio system must support lifecycle management.

---

## 6. The three approved RAG layers

The project includes three distinct RAG layers. They must remain conceptually separate.

## 6.1 Knowledge RAG

### What it stores

- investment rules,
- portfolio construction principles,
- diversification rules,
- risk management rules,
- rebalancing principles,
- asset allocation methodology,
- investment policy guidelines,
- interpretation templates for investor profiles.

### Why it exists
To provide agents with structured investment knowledge that is explicit and retrievable rather than vaguely embedded in the LLM.

### Primary consumers

- Portfolio Construction Agent,
- Risk Agent,
- Critic Agent,
- Explainability Agent.

### Typical usage

- retrieve rules for moderate vs aggressive portfolios,
- retrieve portfolio concentration thresholds,
- retrieve methodology-aligned reasoning templates,
- retrieve rebalancing logic.

---

## 6.2 News / Research RAG

### What it stores

- previously retrieved news,
- prior summaries,
- research notes,
- ticker-level context,
- sector-level context,
- macro commentary,
- extracted event notes.

### Important role definition
This layer is used **primarily as a cache**.

It is not the sole authority on current news. The workflow must be:

1. check News/Research RAG first,
2. decide whether cached content is enough,
3. if stale or incomplete, call tools for fresh retrieval,
4. summarize and normalize the retrieved results,
5. write them back to News/Research RAG.

### Why this design was chosen
Because repeated external searches are slow, expensive, and noisy. This layer reduces redundant tool usage while preserving access to fresh information when needed.

### Primary consumers

- Data Agent,
- Regime Agent,
- Critic Agent,
- Explainability Agent.

### Freshness policy
The system should attach timestamps and freshness metadata to every cached research/news unit so agents can determine whether reuse is acceptable.

---

## 6.3 Decision Memory RAG

### What it stores

- previous portfolios,
- previous weight structures,
- prior critic reports,
- prior risk reports,
- rebalancing decisions,
- monitoring history,
- post-mortem notes,
- approved vs rejected pattern examples.

### Why it exists
It gives the system memory of its own past decisions and helps prevent repeated poor logic.

### Primary consumers

- Critic Agent,
- Monitoring / Rebalancing Agent,
- Explainability Agent.

### Typical usage

- find similar previous cases,
- compare current proposal to prior approved/rejected proposals,
- improve monitoring decisions,
- explain how the current case differs from prior cases.

---

## 7. Overall interaction between Tools, RAG, LLM agents, and ML

This interaction must be respected in implementation.

### 7.1 Tools
Tools are for:

- fresh data retrieval,
- numerical computation,
- optimization,
- backtesting,
- risk metric calculation,
- deterministic operations.

### 7.2 RAG
RAG is for:

- rule retrieval,
- cached context retrieval,
- memory retrieval.

### 7.3 LLM agents
LLM agents are for:

- orchestration,
- interpretation,
- reasoning,
- validation,
- critique,
- explanation.

### 7.4 ML models
ML models are for:

- scoring,
- ranking,
- regime modeling where appropriate,
- predictive signals if used carefully.

### 7.5 Governing principle
A simple way to remember the architecture:

- **tools compute and fetch**,
- **RAG remembers and provides context**,
- **ML scores**,
- **LLM decides how to use all of the above**.

---

## 8. LangGraph orchestration design

LangGraph is the orchestration backbone.

## 8.1 Why LangGraph is used

It is needed because the system is not a single prompt chain. It requires:

- shared state,
- conditional routing,
- retries,
- loops,
- critic-driven revision,
- potentially parallel pre-analysis steps,
- monitoring workflows separate from initial construction.

## 8.2 Base graph flow

The canonical initial graph:

1. `ProfileAgentNode`
2. `DataAgentNode`
3. `FeatureScoringNode`
4. `RegimeNode`
5. `PortfolioConstructionNode`
6. `RiskNode`
7. `CriticNode`
8. conditional branch:
   - if approve → `ExplainabilityNode` → final output
   - if revise_weights → back to `PortfolioConstructionNode`
   - if replace_assets → back to `FeatureScoringNode` or `PortfolioConstructionNode`
   - if reduce_risk → back to `PortfolioConstructionNode`
   - if insufficient_confidence → final output with warning

## 8.3 Monitoring graph flow

A separate graph or subgraph may run later:

1. load current portfolio,
2. refresh data,
3. update scores,
4. update regime,
5. run risk recheck,
6. decide on rebalancing,
7. store outcome in Decision Memory RAG.

---

## 9. State design

A shared typed state must be used instead of ad hoc prompt memory.

## 9.1 Recommended global state structure

```python
state = {
    "user_profile": {},
    "constraints": {},
    "universe": [],
    "market_data": {},
    "fundamentals": {},
    "macro_data": {},
    "news_context": {},
    "features": {},
    "asset_scores": {},
    "regime": {},
    "candidate_portfolio": {},
    "risk_report": {},
    "critic_report": {},
    "final_portfolio": {},
    "explanation": {},
    "decision_log": [],
    "freshness": {},
    "memory_refs": {}
}
```

## 9.2 Why strong state matters

Because every agent should operate on structured shared information, not on brittle conversational assumptions.

## 9.3 Recommended implementation approach

Use typed schemas such as:

- Pydantic models,
- TypedDict / dataclasses,
- schema validation between nodes,
- immutable or controlled updates where possible.

## 9.4 Mitigating state bloat

The global state can easily grow too large (e.g., hundreds of parsed news articles, massive score vectors). To prevent this:

- **Pass only required subsets**: Agents should only receive the keys of the state they genuinely need, not the full state dictionary.
- **Reference pointers**: For heavy data (like large Pandas DataFrames or text chunks), store them in a local storage/DB and only pass the path/ID in the LangGraph state.
- **State trimming**: The graph should periodically flush or clear temporary text blocks (like intermediate LLM reasoning) from the context once they are no longer needed.

---

## 10. Tool map by responsibility

Below is the recommended minimum tool set.

## 10.1 Market data tool

### Used by
- Data Agent,
- Monitoring Agent,
- Risk Agent.

### Purpose
Retrieve asset prices, returns, benchmark history, volumes, and possibly corporate action-adjusted series.

---

## 10.2 Fundamentals tool

### Used by
- Data Agent,
- Feature / Scoring Agent.

### Purpose
Retrieve valuation, profitability, leverage, and quality inputs.

---

## 10.3 News / research retrieval tool

### Used by
- Data Agent,
- Regime Agent,
- Critic Agent,
- Explainability Agent.

### Purpose
Fetch fresh text context when News/Research RAG cache is insufficient or stale.

---

## 10.4 Macro data tool

### Used by
- Data Agent,
- Regime Agent.

### Purpose
Retrieve macroeconomic indicators, yields, inflation proxies, policy rates, or similar regime-relevant context.

---

## 10.5 Feature calculator

### Used by
- Feature / Scoring Agent.

### Purpose
Compute deterministic factor features from normalized market and fundamental data.

---

## 10.6 ML scoring model

### Used by
- Feature / Scoring Agent.

### Purpose
Produce ranking or attractiveness scores for assets.

---

## 10.7 Portfolio optimizer

### Used by
- Portfolio Construction Agent.

### Purpose
Transform candidate assets and constraints into proposed weights.

Possible strategies:

- constrained mean-variance,
- risk-parity style allocation,
- score-weighted capped allocation,
- hybrid optimizer with turnover penalties.

---

## 10.8 Risk metrics calculator

### Used by
- Risk Agent,
- Monitoring Agent.

### Purpose
Compute risk metrics and validate portfolio against policy.

---

## 10.9 Backtesting / simulation tool

### Used by
- research workflows,
- evaluation pipeline,
- optional critic support,
- offline experimentation.

### Purpose
Measure historical behavior of strategies and compare baselines.

---

## 10.10 RAG retriever interface

### Used by
- all agents that need rule/context/memory access.

### Purpose
Retrieve top relevant chunks from one of the three knowledge layers.

---

## 11. Agent-to-RAG usage map

This mapping should be explicit in code.

## 11.1 Profile Agent

### Reads
Usually none, unless policy templates are stored in Knowledge RAG.

### Writes
Usually none.

---

## 11.2 Data Agent

### Reads
- News/Research RAG

### Writes
- News/Research RAG after fresh retrieval and normalization

---

## 11.3 Feature / Scoring Agent

### Reads
Usually none directly, though it may consume processed contextual outputs from Data Agent.

### Writes
Optional experimental logs, not core RAG.

---

## 11.4 Regime Agent

### Reads
- News/Research RAG

### Writes
Optional regime-context notes into News/Research RAG if needed.

---

## 11.5 Portfolio Construction Agent

### Reads
- Knowledge RAG

### Writes
Usually none directly.

---

## 11.6 Risk Agent

### Reads
- Knowledge RAG

### Writes
Optional risk evaluation summary to Decision Memory RAG as part of full case logging.

---

## 11.7 Critic Agent

### Reads
- Knowledge RAG
- News/Research RAG when contradiction analysis needs context
- Decision Memory RAG

### Writes
- critic report into Decision Memory RAG

---

## 11.8 Explainability Agent

### Reads
- Knowledge RAG
- News/Research RAG
- Decision Memory RAG

### Writes
Optional explanation summary into Decision Memory RAG.

---

## 11.9 Monitoring / Rebalancing Agent

### Reads
- Decision Memory RAG
- News/Research RAG if contextual refresh is needed

### Writes
- monitoring outcome to Decision Memory RAG
- updated research cache to News/Research RAG if fresh retrieval occurred

---

## 12. Suggested implementation flow in code

This section tells the coding agent how to build the system in a practical order.

## 12.1 Phase 1 — Define schemas first

Before writing agent logic, define the core typed objects:

- InvestorProfile
- Constraints
- MarketDataBundle
- FeatureSet
- AssetScore
- RegimeReport
- CandidatePortfolio
- RiskReport
- CriticReport
- FinalRecommendation
- MonitoringEvent
- RAGDocument schemas

This reduces chaos later.

---

## 12.2 Phase 2 — Build tools and deterministic services

Implement deterministic services first:

- market data loader,
- fundamentals loader,
- macro data loader,
- feature calculator,
- optimizer,
- risk engine,
- basic storage,
- RAG ingestion and retrieval interfaces.

Reason: the agents are only as good as the underlying tools.

---

## 12.3 Phase 3 — Build RAG layers

### Knowledge RAG
Create and ingest investment methodology documents.

### News/Research RAG
Define cache item schema with:

- entity identifiers,
- summary,
- source timestamp,
- cache timestamp,
- freshness metadata,
- optional confidence.

### Decision Memory RAG
Store case-level documents containing:

- profile summary,
- market regime,
- selected assets,
- weights,
- risk report summary,
- critic verdict,
- final explanation summary,
- outcome metadata.

---

## 12.4 Phase 4 — Implement agents one by one

Recommended order:

1. Profile Agent
2. Data Agent
3. Feature / Scoring Agent
4. Regime Agent
5. Portfolio Construction Agent
6. Risk Agent
7. Critic Agent
8. Explainability Agent
9. Monitoring / Rebalancing Agent

Reason: this follows dependency order.

---

## 12.5 Phase 5 — Assemble LangGraph

Create nodes, wire edges, add typed state transitions, then implement critic-driven loops.

Do not start with a huge graph immediately. First ensure each node behaves correctly in isolation.

---

## 12.6 Phase 6 — Add logging and traceability

Every major decision must be traceable.

Log:

- which tools were called,
- which RAG layer was queried,
- what freshness status was detected,
- what the regime output was,
- what risk warnings were raised,
- what the critic rejected,
- what changed after revision.

This is useful for debugging, evaluation, and thesis demonstration.

---

## 13. Suggested folder structure

A possible production-friendly structure:

```text
project/
  app/
    graph/
      state.py
      nodes.py
      router.py
      workflow.py
    agents/
      profile_agent.py
      data_agent.py
      scoring_agent.py
      regime_agent.py
      portfolio_agent.py
      risk_agent.py
      critic_agent.py
      explainability_agent.py
      monitoring_agent.py
    tools/
      market_data.py
      fundamentals.py
      macro_data.py
      news_retrieval.py
      feature_calc.py
      optimizer.py
      risk_metrics.py
      backtest.py
    rag/
      knowledge_rag.py
      news_cache_rag.py
      decision_memory_rag.py
      ingestion/
      schemas.py
    ml/
      scoring_model.py
      regime_model.py
      training/
    domain/
      schemas.py
      constants.py
      constraints.py
    storage/
      db.py
      repositories/
    api/
      routes.py
      dto.py
    ui/
      streamlit_app.py
  data/
  notebooks/
  tests/
  README.md
```

This exact structure can change, but the separation of concerns must remain.

---

## 14. Freshness and caching strategy

Freshness logic matters because the News/Research RAG is explicitly a cache.

## 14.1 Freshness rules

Each research/news object should include:

- source publication time,
- retrieval time,
- summarization time,
- entity mapping,
- freshness TTL or staleness classification.

## 14.2 Retrieval logic

When an agent requests news context:

1. retrieve cached entries,
2. inspect coverage and freshness,
3. if sufficient → use cache,
4. if insufficient → fetch fresh results with tools,
5. summarize,
6. store back into RAG,
7. continue workflow.

## 14.3 Why this matters

Without freshness logic, a cache quickly becomes misleading and harms decisions.

---

## 15. Critic loop behavior

This is one of the most important sections of the system.

## 15.1 Goal
The critic is not decorative. It is a control mechanism.

## 15.2 What the critic must inspect

- score-to-weight consistency,
- concentration vs profile,
- contradiction with regime,
- contradiction with risk report,
- excessive reliance on weak-confidence assets,
- repeated historically poor patterns,
- lack of justification,
- suspiciously overfit-looking proposals.

## 15.3 What happens after rejection

The system must branch based on rejection reason.

### Example routing
- if issue is concentration → return to Portfolio Construction Agent with stricter caps
- if issue is weak asset set → return to Feature / Scoring Agent or Portfolio Construction Agent
- if issue is missing confidence → allow final output with warning or request more data
- if issue is stale context → call Data Agent to refresh text context

## 15.4 What must be stored
Every critic decision should be saved to Decision Memory RAG.

## 15.5 Mitigating infinite critic loops

Because the Critic Agent can repeatedly reject the Portfolio Construction Agent’s proposals, the graph can easily get stuck in an infinite loop ("propose → reject → propose same → reject same"). To prevent this:

- **Circuit breaker**: Implement a hard limit on routing back to revision (e.g., `MAX_REVISIONS = 3`).
- **Forced decision**: If the limit is reached, the system must forcefully branch to either output the best-found portfolio (with a stern risk warning) or declare a failure to synthesize a valid portfolio.
- **Stateful critique history**: Pass the `critic_report` of *previous* failed attempts of the same session back into the Portfolio Construction Agent, so it explicitly knows what it already tried and why it failed.

---

## 16. Monitoring and rebalancing logic

## 16.1 Why it exists
A portfolio system without monitoring is incomplete.

## 16.2 What should trigger monitoring decisions

- periodic review,
- weight drift,
- regime shift,
- sudden increase in volatility,
- change in score ranking,
- risk threshold breach.

## 16.3 What monitoring should output

- hold,
- rebalance now,
- reduce risk,
- reassess universe,
- escalate for manual review.

## 16.4 Memory integration
Monitoring outcomes must be written back into Decision Memory RAG for future retrieval.

---

## 17. What this project is explicitly not

To keep implementation aligned, the coding agent must avoid drifting into the wrong design.

### It is not

- a single super-agent that does everything,
- a chatbot that improvises financial advice,
- a pure price-forecasting project,
- a sentiment-only stock picker,
- an LLM-generated optimizer,
- a system where RAG replaces market data tools.

---

## 18. Recommended evaluation directions

Even if evaluation is implemented later, the architecture should support it.

## 18.1 Financial metrics

- cumulative return,
- annualized return,
- volatility,
- Sharpe ratio,
- Sortino ratio,
- max drawdown,
- Calmar ratio.

## 18.2 Portfolio quality metrics

- concentration,
- diversification score,
- average pairwise correlation,
- turnover,
- stability of weights.

## 18.3 Agent-system metrics

- critic rejection rate,
- average revision loops before approval,
- percentage of cases where critic reduced risk,
- cache hit ratio for News/Research RAG,
- tool call reduction due to cache usage,
- number of monitoring-triggered rebalances.

---

## 19. Implementation priorities for the coding agent

If time is limited, implement in this order:

1. typed schemas,
2. deterministic tools,
3. Knowledge RAG,
4. News/Research RAG cache,
5. scoring pipeline,
6. portfolio construction,
7. risk validation,
8. critic loop,
9. explainability,
10. decision memory,
11. monitoring and rebalancing.

This order is more stable than trying to build all agents at once.

---

## 20. Final architectural summary

The final approved architecture is:

- **LangGraph-based multi-agent decision-support system**
- main pipeline:
  - Profile Agent
  - Data Agent
  - Feature / Scoring Agent
  - Regime Agent
  - Portfolio Construction Agent
  - Risk Agent
  - Critic Agent
  - Explainability Agent
- separate lifecycle branch:
  - Monitoring / Rebalancing Agent
- core decision logic:
  - multi-factor scoring
  - regime adaptation
  - constrained portfolio optimization
  - critic validation
- three RAG layers:
  - Knowledge RAG
  - News/Research RAG used primarily as cache
  - Decision Memory RAG
- division of labor:
  - tools fetch and compute,
  - ML scores,
  - RAG provides context and memory,
  - LLM agents orchestrate, critique, and explain.

This summary is the project’s authoritative architectural baseline.

---

## 21. Direct instructions for the coding AI agent

When implementing this system, follow these rules:

1. Never let the LLM directly produce trusted numerical portfolio weights without a deterministic calculation step.
2. Keep each agent focused on one responsibility.
3. Use typed shared state across LangGraph nodes.
4. Use Knowledge RAG for rules, not as a generic document dump.
5. Use News/Research RAG as a cache first, then tools if stale or incomplete.
6. Always write back fresh processed research context to the cache after tool-based retrieval.
7. Store critic reports and monitoring outcomes into Decision Memory RAG.
8. Keep risk validation independent from portfolio construction.
9. Do not skip the critic loop.
10. Make every major decision inspectable through logs, reports, and stored artifacts.
