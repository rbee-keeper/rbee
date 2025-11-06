# SEO + COPY AUDIT (rbee Commercial Frontend)

Date: 2025-11-06
Scope: All .tsx PageProps and related routing under frontend/apps/commercial
Perspective: Google Helpful Content + E-E-A-T. No rewrites; diagnosis and structural recommendations only.

# Part 1 — Core Funnel Pages

## HomePage
Purpose: Primary landing; explain rbee and route to personas/features.
Intent: Navigational + Informational (with Commercial investigation cues).
Primary keywords: self-hosted OpenAI-compatible API; multi-machine GPU orchestration; heterogeneous hardware.
Secondary keywords: homelab LLM; SSH deployment; Rhai policy routing; GDPR-ready AI.
Clarity test: Pass — “Use ALL Your GPUs with One API.” (≤10 words, concrete).
Differentiation: Strong. Presents 6 unique advantages (multi-machine; CUDA+Metal+CPU; SSH-only; Rhai scripts; GDPR module; lifetime pricing). Clear contrast vs Ollama/vLLM/Together/Ray.
Duplication risk: Moderate with FeaturesPage and HomelabPage. Keep as canonical “why” + overview; push detail to feature pages via internal links (already present partly).
AI-tone severity: Low–Medium. Some hype (“Free forever”, metaphor “queen/colony”). Generally concrete.
Authority gap: Claims like “87% utilization” and “5-minute setup” lack sources. Add benchmark and case-study links; cite repos/tests; add lightweight schema (FAQ/HowTo) and link to technical deep-dive.
SEO value score: 8/10
Recommended action: Keep. Add cite-able proof (benchmarks/case study); ensure internal links to Feature detail pages and Comparisons.

## PricingPage
Purpose: Pricing and monetization narrative (lifetime vs subscription) + tiers.
Intent: Commercial + Transactional.
Primary keywords: rbee pricing; lifetime pricing; GDPR auditing pricing; premium features pricing.
Secondary keywords: self-hosted AI costs; OpenAI alternative pricing; one-time license; ROI.
Clarity test: Pass — “Free Forever. Premium Optional.” (clear value frame).
Differentiation: Strong lifetime pricing vs cloud API subscriptions; cost comparison table adds clarity.
Duplication risk: Moderate with StartupsPage cost narrative; minor with comparisons (Together.ai). Avoid duplicate ROI copy; canonicalize cost math to Pricing.
AI-tone severity: Medium (marketing badges). Still mostly specific.
Authority gap: “€499 vs $72K/year” requires assumptions. Add visible methodology note, calculators, scenario presets; add Product schema for SKUs.
SEO value score: 8/10
Recommended action: Keep. Canonical home for “cost/ROI” keywords; link out to Startups for persona story and to comparisons for provider-specific costs.

## FeaturesPage
Purpose: Feature hub; summary + interactive tabs.
Intent: Informational (Commercial investigation).
Primary keywords: rbee features; OpenAI-compatible API; multi-GPU orchestration; programmable scheduler; SSE.
Secondary keywords: heterogeneous hardware; Rhai scripting; error handling; model management.
Clarity test: Pass — “Core capabilities” + tab labels are concrete.
Differentiation: Strong (Rhai, SSH, heterogeneous mix, SSE timeline). Code examples add specificity.
Duplication risk: High with Feature detail pages (OpenAI-Compatible, Multi-Machine, Rhai, Heterogeneous). Use Features as hub; detail pages as canonical for long-tail.
AI-tone severity: Low. Technical, example-led.
Authority gap: Add performance deltas (latency/throughput) and link to docs/specs; include structured data (FAQ) on common feature questions.
SEO value score: 8/10
Recommended action: Keep. Make it a hub with prominent “Learn more” deep links (already in props) and add FAQs.

## UseCasesPage
Purpose: Use-case hub (personas + industries).
Intent: Informational (Top-of-funnel exploration).
Primary keywords: LLM use cases; OpenAI-compatible use cases; industry AI use cases.
Secondary keywords: homelab; enterprise; research; finance/healthcare/legal AI.
Clarity test: Pass — explains independence and who benefits.
Differentiation: Moderate; breadth over depth.
Duplication risk: High with persona and industry pages (Developers, Homelab, Enterprise, Legal). Make this an index/hub only; avoid repeating long text.
AI-tone severity: Low–Medium.
Authority gap: Needs outbound links to detailed case pages; add schema for breadcrumb/ItemList.
SEO value score: 6.5/10
Recommended action: Keep as index. Convert heavy copy to concise cards that deep-link to canonical persona/industry pages.

# Part 2 — Personas & Industries

## DevelopersPage
Purpose: Persona page for devs adopting OpenAI-compatible local stack.
Intent: Commercial investigation.
Primary keywords: self-hosted AI for developers; OpenAI-compatible local API.
Secondary keywords: Zed/Cursor configuration; SSE; code generation.
Clarity test: Pass — “Build with AI. Own your infrastructure.”
Differentiation: Good developer-first framing, concrete code.
Duplication risk: Medium with OpenAI-Compatible page and Features hub.
AI-tone severity: Low–Medium.
Authority gap: Add reproducible example repos; a short quickstart (curl/Zed/Cursor) and link to docs.
SEO value score: 7/10
Recommended action: Keep. Narrow to dev-proof (snippets, config examples); defer cost talk to Pricing.

## HomelabPage
Purpose: Persona/industry for homelab users.
Intent: Informational + Commercial.
Primary keywords: homelab LLM; self-host OpenAI API; multi-machine homelab.
Secondary keywords: CUDA+Metal+CPU; SSH deployment; LAN-only.
Clarity test: Pass — “One API for All.” and “Stop Juggling ComfyUI + Ollama + Whisper.”
Differentiation: Strong; practical steps.
Duplication risk: Medium with Multi-Machine feature and Developers.
AI-tone severity: Medium (punchy copy). Balanced by specifics.
Authority gap: Add real homelab topology example with hardware list and perf; link to setup docs.
SEO value score: 8/10
Recommended action: Keep. Link to Multi-Machine + Heterogeneous feature pages; add a real build log/bench.

## EnterprisePage
Purpose: B2B page for regulated orgs.
Intent: Commercial investigation (B2B lead-gen).
Primary keywords: GDPR-compliant AI infrastructure; EU data residency AI; SOC2 ISO 27001 AI.
Secondary keywords: audit logs; compliance endpoints; EU-only deployment.
Clarity test: Pass — “AI Infrastructure That Meets Your Compliance Requirements.”
Differentiation: Strong compliance-first positioning.
Duplication risk: High with CompliancePage and SecurityPage. One should be canonical; others index/support.
AI-tone severity: Low–Medium.
Authority gap: Add compliance whitepaper, auditor quotes, and DPA templates; more substantiation for “Zero US cloud deps”.
SEO value score: 7.5/10
Recommended action: Keep. Make Enterprise the sales/solutions page; push compliance details to CompliancePage (canonical for standards), technical to SecurityPage.

## EducationPage
Purpose: Education persona (courses/universities).
Intent: Informational.
Primary keywords: distributed systems education; hands-on AI labs; open-source curriculum.
Secondary keywords: BDD testing; SSE; multi-GPU labs.
Clarity test: Pass — “Teach Distributed AI with Real Infrastructure.”
Differentiation: Good; curriculum framing.
Duplication risk: Medium with ResearchPage (both academic). Distinct if Education = teaching, Research = reproducibility.
AI-tone severity: Low–Medium.
Authority gap: Cite universities, syllabi, GitHub orgs; publish sample syllabus PDFs.
SEO value score: 6.5/10
Recommended action: Keep. Add real university/social proof and curriculum assets.

## ResearchPage
Purpose: Research persona (reproducibility/determinism).
Intent: Informational + Commercial.
Primary keywords: reproducible AI; deterministic seeds; proof bundles; experiment tracking.
Secondary keywords: multi-modal deterministic; verification.
Clarity test: Pass — “Reproducible AI for Scientific Research.”
Differentiation: Strong; unique “proof bundles.”
Duplication risk: Low–Medium with Education. Complementary if scoped.
AI-tone severity: Low.
Authority gap: Publish method paper or blog with metrics; link to determinism suite docs.
SEO value score: 7.5/10
Recommended action: Keep. Add citations and downloadable examples.

## StartupsPage
Purpose: Persona for startup economics.
Intent: Commercial investigation.
Primary keywords: self-hosted AI for startups; OpenAI cost savings; per-token savings.
Secondary keywords: MVP to scale; multi-GPU cluster.
Clarity test: Pass — “Escape API Fees.”
Differentiation: Good economic framing; overlaps with Pricing.
Duplication risk: High with Pricing; some with comparisons (Together.ai).
AI-tone severity: Medium.
Authority gap: Needs calculator defaults and case studies; point cost math to Pricing canonical.
SEO value score: 7/10
Recommended action: Keep. Reduce overlapping math; link to Pricing and Together comparison.

## ProvidersPage
Purpose: Provider acquisition (supply side of marketplace).
Intent: Transactional lead-gen.
Primary keywords: monetize GPU; rent GPU compute; GPU marketplace.
Secondary keywords: commission 15%; earnings calculator; sandboxed jobs.
Clarity test: Pass — “Your GPUs Can Pay You Every Month.”
Differentiation: Clear commission and security promises.
Duplication risk: Low.
AI-tone severity: Medium (income claims). Provide caveats (present but expand methodology).
Authority gap: Add real provider stories, payout screenshots (anonymized), energy cost model.
SEO value score: 7/10
Recommended action: Keep. Tighten assumptions in calculator; add disclaimers and examples.

## Legal (Industry) Page
Purpose: Industry vertical (law firms).
Intent: Commercial investigation.
Primary keywords: legal AI on-prem; attorney-client privilege AI; legal research AI.
Secondary keywords: audit trail legal; DMS integration; citation verification.
Clarity test: Pass — strong headline.
Differentiation: Strong legal-specific controls.
Duplication risk: Low–Medium with Compliance/Security; industry-specific context is unique.
AI-tone severity: Low–Medium.
Authority gap: Add case study and partner logos; clarify insurer/ethics citations.
SEO value score: 7/10
Recommended action: Keep. Add case study PDF + schema (Article/FAQ).

# Part 3 — Feature Detail Pages

## OpenAICompatiblePage
Purpose: Deep feature page for drop-in compatibility.
Intent: Informational + Commercial.
Primary keywords: OpenAI-compatible API; OpenAI alternative self-hosted.
Secondary keywords: Zed/Cursor config; SDK example.
Clarity test: Pass — “Drop-In Replacement for OpenAI.”
Differentiation: Clear and technical.
Duplication risk: High with DevelopersPage; make this canonical for “OpenAI-compatible” keyword; Developers should link here.
AI-tone severity: Low.
Authority gap: Add compatibility matrix (endpoints/SDKs) and pitfalls.
SEO value score: 8/10
Recommended action: Keep. Make canonical for “OpenAI-compatible” cluster.

## MultiMachinePage
Purpose: Deep feature page for multi-machine orchestration.
Intent: Informational + Commercial.
Primary keywords: multi-machine LLM orchestration; SSH deployment LLM.
Secondary keywords: load balancing; failover; horizontal scale.
Clarity test: Pass — “Use ALL Your GPUs Across ALL Your Machines.”
Differentiation: Strong; anti-Kubernetes stance.
Duplication risk: Medium with Homelab page; but this is feature-depth. Canonical for “multi-machine orchestration”.
AI-tone severity: Low.
Authority gap: Add benchmark vs single-machine; include topology diagrams perf.
SEO value score: 8/10
Recommended action: Keep. Add proof (numbers) + schema.

## HeterogeneousHardwarePage
Purpose: Deep feature page for CUDA+Metal+CPU mix.
Intent: Informational.
Primary keywords: heterogeneous hardware LLM; mix NVIDIA Apple AMD.
Secondary keywords: backend selection; error handling; detection.
Clarity test: Pass — “CUDA + Metal + CPU Together.”
Differentiation: Strong.
Duplication risk: Medium with Features and Homelab descriptions.
AI-tone severity: Low.
Authority gap: Add support matrix table; real detection logs; device counts.
SEO value score: 8/10
Recommended action: Keep. Ensure the .bak variant is removed or redirected to avoid duplicate content in repo history.

## RhaiScriptingPage
Purpose: Deep feature page for policy routing.
Intent: Informational.
Primary keywords: Rhai routing; policy-based LLM routing.
Secondary keywords: A/B testing with routing; GDPR routing.
Clarity test: Pass — “User-Scriptable Routing.”
Differentiation: Strong and unique.
Duplication risk: Low–Medium with Features tab and Developers.
AI-tone severity: Low.
Authority gap: Add real scripts (repo), performance/cost impact examples.
SEO value score: 7.5/10
Recommended action: Keep. Add sample scripts library + doc links.

# Part 4 — Comparison Pages

## RbeeVsOllamaPage
Purpose: Buyer comparison (single vs multi-machine).
Intent: Commercial investigation.
Primary keywords: rbee vs ollama; Ollama multi-machine alternative.
Secondary keywords: multi-GPU; heterogeneous hardware; redundancy.
Clarity test: Pass.
Differentiation: Clear decision matrix.
Duplication risk: Low within comparison cluster.
AI-tone severity: Low.
Authority gap: Add citations/screens from Ollama docs for fairness; link to features.
SEO value score: 7.5/10
Recommended action: Keep. Add “when to choose” JSON-LD FAQ.

## RbeeVsVllmPage
Purpose: Buyer comparison (SSH vs Kubernetes complexity).
Intent: Commercial investigation.
Primary keywords: rbee vs vllm; vllm Kubernetes alternative.
Secondary keywords: heterogeneous hardware; Apple Silicon support.
Clarity test: Pass.
Differentiation: Clear deployment contrast.
Duplication risk: Low.
AI-tone severity: Low.
Authority gap: Provide vLLM docs links and caveats; be precise about device support status.
SEO value score: 7.5/10
Recommended action: Keep. Add table with source links.

## RbeeVsRayKservePage
Purpose: Buyer comparison (SSH vs Ray+KServe DevOps complexity).
Intent: Commercial investigation.
Primary keywords: rbee vs ray kserve; ray serve alternative without kubernetes.
Secondary keywords: setup time; DevOps team requirement.
Clarity test: Pass.
Differentiation: Strong.
Duplication risk: Low.
AI-tone severity: Low.
Authority gap: Cite Ray/KServe docs for setup steps/time; avoid overclaim.
SEO value score: 7.5/10
Recommended action: Keep. Add step list with sources.

## RbeeVsTogetherAiPage
Purpose: Buyer comparison (cost/privacy).
Intent: Commercial investigation.
Primary keywords: rbee vs together.ai; OpenAI/Together cost comparison.
Secondary keywords: $/tokens; annual cost.
Clarity test: Pass.
Differentiation: Strong cost/privacy frame.
Duplication risk: Medium with Pricing/Startups; centralize math in Pricing and reference here.
AI-tone severity: Medium.
Authority gap: Provide cost calculator assumptions and ranges; sensitivity analysis.
SEO value score: 7/10
Recommended action: Keep. Link pricing calculator; add methodology box.

# Part 5 — Compliance, Security, Community, Legal Docs

## CompliancePage
Purpose: Standards/compliance hub; deep content.
Intent: Informational (B2B lead-gen assist).
Primary keywords: GDPR AI compliance; SOC2 AI; ISO 27001 AI.
Secondary keywords: audit trail; data residency; retention.
Clarity test: Pass.
Differentiation: Strong.
Duplication risk: High with Enterprise and Security. Make this canonical for compliance standards; Enterprise = solution; Security = architecture.
AI-tone severity: Low.
Authority gap: Add downloadable compliance pack; references to legal texts/DPAs; auditor quotes.
SEO value score: 7.5/10
Recommended action: Keep. Canonicalize “compliance” cluster here.

## SecurityPage
Purpose: Security architecture and guarantees.
Intent: Informational (technical due diligence).
Primary keywords: AI security architecture; zero trust AI; immutable audit logs.
Secondary keywords: constant-time; zeroization; JWT; deadlines.
Clarity test: Pass.
Differentiation: Strong.
Duplication risk: Medium with Compliance; technical angle is distinct.
AI-tone severity: Low.
Authority gap: Add links to crates and security docs; add threat model PDF; CVE policy.
SEO value score: 7/10
Recommended action: Keep. Link to crate docs and security audit results when available.

## CommunityPage
Purpose: OSS community growth.
Intent: Informational.
Primary keywords: rbee community; open-source AI community.
Secondary keywords: GitHub stars; Discord.
Clarity test: Pass.
Differentiation: Standard.
Duplication risk: Low.
AI-tone severity: Low–Medium (vanity stats, fine).
Authority gap: Add contributor spotlights and links to “good first issues”.
SEO value score: 5.5/10
Recommended action: Keep minimal; do not chase SEO here; ensure internal links.

## PrivacyPage
Purpose: Legal privacy policy.
Intent: Informational (compliance).
Primary keywords: privacy policy; GDPR rights; data processing.
Secondary keywords: data residency; telemetry; cookies.
Clarity test: Pass.
Differentiation: N/A.
Duplication risk: Low.
AI-tone severity: Low.
Authority gap: Add DPA link; controller/processor roles; contact mailbox (present) and jurisdiction info (present).
SEO value score: 5/10
Recommended action: Keep. Ensure canonical /legal/privacy; add ToC and anchors.

## TermsPage
Purpose: Terms of service.
Intent: Informational (compliance).
Primary keywords: terms of service; GPL license terms; acceptable use.
Secondary keywords: dispute resolution; governing law.
Clarity test: Pass.
Differentiation: N/A.
Duplication risk: Low.
AI-tone severity: Low.
Authority gap: None critical.
SEO value score: 5/10
Recommended action: Keep. Ensure last-updated date markup.

# Global Summary

- **Keyword gaps and overlaps**
  - Gaps: “download rbee” landing; “installation guide” hub; “benchmarks”/“performance” pages; “case studies” per persona; “model support matrix” (CUDA/Metal/CPU table) canonical.
  - Overlaps: Cost narrative (Pricing vs Startups vs Together.ai); OpenAI-compatible (Developers vs OpenAI-Compatible); Compliance (Enterprise vs Compliance vs Security); Multi-machine (Homelab vs Multi-Machine feature).

- **Canonical page recommendations**
  - Home: brand + positioning (canonical “rbee” + “self-hosted OpenAI-compatible” intent).
  - Pricing: canonical for cost/ROI; other pages link to it for math claims.
  - Features: hub only; canonical detail pages: OpenAI-Compatible, Multi-Machine, Heterogeneous, Rhai.
  - Comparison cluster: keep each “/compare/*” page; add a “/compare” hub index.
  - Compliance cluster: Compliance (standards hub) canonical; Enterprise (solutions/sales), Security (technical) support it.
  - Personas: Developers, Homelab, Startups, Research, Education, Providers, Legal — keep, but avoid duplicating feature copy; link to canonical feature pages.

- **Suggested consolidation tree**
  - Cost/ROI: Startups (persona story) → Pricing (canonical math) → Together.ai comparison (provider-specific).
  - OpenAI-compatible: Developers → OpenAI-Compatible (canonical) → Features hub.
  - Multi-machine: Homelab → Multi-Machine (canonical) → Heterogeneous.
  - Compliance: Enterprise → Compliance (canonical) → Security.
  - Use Cases: act as index linking into personas/industries; reduce body copy.

- **Domain-level keyword focus strategy**
  - Primary clusters (own these SERPs):
    - “self-hosted OpenAI-compatible API” (OpenAI-Compatible + Developers + Home)
    - “multi-machine GPU orchestration” (Multi-Machine + Homelab + Features)
    - “heterogeneous hardware (CUDA + Metal + CPU)” (Heterogeneous + Features)
    - “GDPR-compliant AI infrastructure / EU data residency AI” (Compliance + Enterprise + Security)
  - Secondary clusters:
    - “Rhai routing / policy-based LLM routing” (Rhai page + docs/examples)
    - “GPU provider marketplace / monetize GPU” (Providers)
    - “reproducible AI research (deterministic seeds)” (Research)

- **Ranking of highest-ROI pages**
  1. Pricing (8/10) — intent-aligned, high conversion if math is credible
  2. OpenAI-Compatible (8/10) — strong query-fit, dev intent
  3. Home (8/10) — brand + broad intent, must link well
  4. Multi-Machine (8/10) — unique differentiator, great long-tail
  5. Features (8/10) — hub boost for internal linking
  6. Compliance (7.5/10) — B2B decision-maker queries
  7. Rbee vs Ollama (7.5/10) — high-intent comparison
  8. Heterogeneous Hardware (8/10) — defensible niche
  9. Providers (7/10) — supply acquisition
  10. Research (7.5/10) — authority building; long-term traffic

- **Overall site readiness score**: 74/100
  - Strengths: Clear value, unique differentiators, decent IA, many templates with FAQs, internal linking present.
  - Weaknesses: Insufficient citations/benchmarks; duplication across personas/features; cost claims need methodology; lack of structured data on key pages; missing hub pages (compare index, download/installation, benchmarks/case studies).

# Priority Fixes (next 2 weeks)
- **Add proof and methodology**
  - Publish a lightweight “Benchmarks & Methodology” page; cite utilization and setup time; link from Home, Features, Multi-Machine.
  - Add Pricing calculator assumptions + sensitivity sliders; centralize math here; ensure comparisons point back here.
- **Canonicalize clusters**
  - Make Compliance the standards hub; Enterprise (solutions) and Security (technical) link inward.
  - Make OpenAI-Compatible the canonical for compatibility keyword; Developers page links inward and focuses on dev workflows.
  - Make Multi-Machine canonical for orchestration; Homelab page focuses on scenarios and links inward.
- **Structured data**
  - Add JSON-LD: FAQ (Pricing, Comparisons, Compliance), Product/Offer (Pricing tiers), HowTo (Install quickstart), Breadcrumb across hubs.
- **Internal linking and nav**
  - Add “/compare” hub, list all comparisons; ensure cross-links from Features and Pricing.
  - From Home hero CTAs, add secondary deep links to OpenAI-Compatible and Multi-Machine.
- **Add case studies**
  - At least 2 persona-aligned case studies (Homelab build log; SME team migrating from OpenAI) with metrics.

# Notes on Routing and Content Hierarchy
- components/pages/index.ts confirms available page set; ensure Next.js app routes under /app/* map to these.
- Comparison routes exist under /compare/* — add /compare index page to avoid orphaned cluster.

— End of analysis —
