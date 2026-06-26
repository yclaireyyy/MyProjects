# Product Demonstration

## Overview

This page documents the product demonstration for the **AI Research Commercialisation Workbench**.

The demonstration presents a working prototype that supports users in moving from an early research idea to a clearer commercialisation opportunity. The product helps users analyse research projects, refine the analysis through follow-up questions, discover related opportunities, and review the final Opportunity Brief before export.

The demonstration focuses on the core user journey: opportunity input, multi-agent analysis, workspace-based refinement, pre-export quality review, and PDF export.

---

## Demonstration Purpose

The purpose of this product demonstration is to show that the prototype can support the main product workflow end-to-end:

1. A user inputs a research project, technology idea, or discovery lead.
2. The system generates a structured four-dimension commercialisation analysis.
3. The system synthesises specialist analysis into an Opportunity Brief.
4. The user can ask follow-up questions to refine the analysis.
5. The user can control the conversation context by ignoring selected responses.
6. The user can apply useful follow-up insights back into the brief.
7. The user can run a pre-export quality check before downloading the final Opportunity Brief.

This demonstration shows that the prototype is not only a one-time AI summary tool, but a structured workflow for investigation, refinement, evidence-aware review, and decision support.

---

## Live Product

**Live demo:**
[https://dt-koala-demo.streamlit.app](https://dt-koala-demo.streamlit.app)

Access is currently not open to external viewers. The walkthrough below documents the product experience for review.

---

## Product Scope Demonstrated

| Area | Description |
|---|---|
| Single Project Analysis | Allows users to input a research project and generate a structured commercialisation analysis |
| Multi-agent Analysis Workflow | Uses specialist agents to assess research understanding, market fit, timing signals, and feasibility |
| Synthesis Agent | Consolidates specialist outputs into a structured Opportunity Brief |
| Analysis Workspace | Gives users a central workspace for reviewing, questioning, and refining the brief |
| Follow-up Conversation | Allows users to ask follow-up questions within the same project context |
| Context Control | Allows users to ignore selected AI responses from future context without deleting them |
| Apply to Brief | Allows useful follow-up insights to update the structured Opportunity Brief |
| Pre-export Quality Check | Reviews the brief before export and highlights evidence gaps, weak claims, unresolved validation needs, and overconfident scoring |
| PDF Export | Allows users to download the refined Opportunity Brief |

---

## Demonstration Walkthrough

### 1. Landing Page And Main Workflow

The demonstration starts from the main application interface.

The user is presented with two main workflows:

- **Analyze a Project**
- **Discover Opportunities**

The **Analyze a Project** workflow is used when the user already has a specific research project or technology idea. The **Discover Opportunities** workflow is used when the user starts from a broader keyword, topic, or strategic domain.

This structure reflects the intended product experience: users can either analyse a known project or explore possible opportunities from a broader area.

---

### 2. Single Project Analysis

The first demonstrated feature is the single project analysis workflow.

The user enters a research project title or technology idea. Optional links or source information can also be provided where available.

After submission, the system generates a structured commercialisation analysis across four dimensions:

1. **Project Understanding**
2. **Industry / Market Analysis**
3. **Signals / Timing**
4. **Feasibility / Commercialisation Potential**

The system also displays an overall summary, key strengths, key risks, and recommended next steps. This helps the user quickly understand the opportunity, its potential value, and its main uncertainties.

---

### 3. Multi-agent Analysis And Synthesis

The product uses a multi-agent workflow rather than a single generic response.

Specialist agents examine different parts of the opportunity:

- The Research Agent focuses on technical framing and research understanding.
- The Market Agent focuses on possible applications, customer problems, and market logic.
- The Timing Agent focuses on external signals such as policy, funding, adoption pressure, or industry activity.
- The Feasibility Agent focuses on development pathway, commercialisation practicality, and validation needs.

The Synthesis Agent then consolidates these specialist outputs into a structured Opportunity Brief. This makes the result easier to inspect, refine, and use as a decision-support artifact.

---

### 4. Analysis Workspace

After the initial analysis is generated, the result is displayed in a dedicated analysis workspace.

The workspace allows the user to review the generated analysis in a structured format. Instead of displaying one long unorganised AI response, the output is separated into meaningful sections.

This workspace acts as the central place where the user can continue working on the same opportunity.

---

### 5. Follow-up Questioning

The demonstration then shows the follow-up question feature.

After reviewing the initial brief, the user can ask further questions about the same project. For example, the user may ask about:

- The likely customer segment
- Main commercial risks
- Potential competitors
- Market timing
- Feasibility concerns
- Possible next validation steps

The Workspace Hub uses the existing project context and routes the question to the most relevant response pathway. This helps the user refine the opportunity without repeating the same project information.

---

### 6. Context Control And Brief Updates

The demonstration also shows the user's control over AI outputs.

The user can ignore a specific AI response. When a response is ignored, it remains visible in the conversation history but is excluded from later context. This improves transparency while helping the user keep future analysis aligned with useful information.

The user can also apply a useful response back into the Opportunity Brief. This connects conversational refinement with the structured output, so the brief can evolve as the user develops a clearer understanding of the opportunity.

---

### 7. Pre-export Quality Check

Before exporting the Opportunity Brief, the demonstration shows the pre-export quality check.

This feature acts as a lightweight decision-readiness review. It does not simply check formatting or grammar. Instead, it checks whether the brief is safe to rely on before it becomes a shareable PDF.

The pre-export check reviews whether:

- Important claims are supported by enough evidence
- Some sections still depend on assumptions or hypotheses
- Market, timing, or feasibility signals are missing or deferred
- The opportunity score may sound more confident than the evidence supports
- Key risks and counterarguments are explicit enough for decision-making

After the user clicks **Run final review**, the system returns a readiness label, a quality score, and evidence checks. For example, it may suggest validating customer demand, resolving an unresolved timing signal, softening an unsupported claim, or clarifying feasibility evidence.

This step is useful because early-stage research commercialisation often involves uncertainty. The pre-export check helps users avoid presenting an early Opportunity Brief as more certain than it really is.

---

### 8. Opportunity Discovery

The second major workflow demonstrated is opportunity discovery.

The user enters a keyword, topic, or strategic domain. The system then generates a list of recommended opportunities. Each recommendation includes a short title, summary, source context, application angle, evidence basis, and suggested next step.

This workflow is useful when the user does not yet have a specific project in mind and wants to explore possible commercialisation directions.

---

## 1.0 vs 2.0 Scope

The demonstration shows the 1.0 product scope: a working end-to-end workflow for research opportunity input, multi-agent analysis, workspace-based refinement, pre-export review, and PDF export.

The broader 2.0 upgrade is planned across four productization tracks: AI output quality governance, evaluation automation, model/cost governance, and context/memory safety. The current active work is the AI quality layer: making recurring AI risks measurable by turning unsupported claims, overconfident recommendations, context loss, and overly broad brief updates into fixed evaluation scenarios.

These scenarios support automated regression-style evaluation when prompts, agents, model routing, or brief update logic changes. Human review remains the decision checkpoint for high-risk outputs and externally shared Opportunity Briefs.

---

## Summary

Overall, the prototype demonstrates a working end-to-end experience from early project input to structured commercialisation decision support, including multi-agent analysis, synthesis, iterative refinement, pre-export quality review, opportunity discovery, and PDF export.
