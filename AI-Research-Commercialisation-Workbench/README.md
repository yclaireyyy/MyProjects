# AI Research Commercialisation Workbench

An AI-native decision-support workflow that helps turn early research signals into structured commercialisation briefs.

**Live demo:** https://dt-koala-demo.streamlit.app (access is currently not open to external viewers; please refer to the documentation below for a walkthrough)
**Role:** AI Product Manager / AI Product Engineer
**Status:** Working prototype / public showcase

## Overview

The AI Research Commercialisation Workbench is a prototype for research commercialisation and early venture opportunity assessment. It helps users move from an early research idea or discovery lead to a structured Opportunity Brief that can be refined through follow-up questions and reviewed before export.

The product is designed for scenarios where research information is fragmented, commercial potential is difficult to assess manually, and users need a clearer way to connect technical evidence, market reasoning, timing signals, and feasibility considerations.

## Highlights

- **Discovery-to-Brief Workflow:** supports a user journey from research opportunity input or discovery lead to a structured commercialisation brief.
- **Multi-agent Analysis Workflow:** coordinates Research, Market, Timing, Feasibility, and Synthesis agents to produce a four-dimension opportunity assessment.
- **Synthesis Agent for Opportunity Briefs:** consolidates specialist agent outputs into a structured, evidence-aware Opportunity Brief instead of a raw AI summary.
- **Workspace Hub:** routes follow-up questions, manages shared context, and supports iterative refinement through Retry, Ignore, and Apply to Brief.
- **Pre-export Quality Check:** reviews the brief before export and highlights evidence gaps, weak claims, unresolved validation needs, and overconfident scoring.
- **Workspace UI Improvements:** improves structured brief display, follow-up interaction, agent-source transparency, and review/export flow.

## My Contribution

- Redesigned the AI workflow from a linear multi-GPT flow into a workspace-based multi-agent system.
- Designed and implemented the Synthesis Agent workflow for structured Opportunity Brief generation.
- Designed the Workspace Hub orchestration layer for intent-aware routing, context management, Retry, Ignore, and Apply-to-Brief.
- Contributed to product architecture and user journey design across input, analysis, refinement, quality review, and export.
- Designed and implemented the Pre-export Quality Check.
- Improved key workspace UI flows, including structured brief display, follow-up controls, agent-source transparency, brief update feedback, and final review placement.

## Current Upgrade Direction

The working prototype represents the 1.0 product scope: an end-to-end workflow for moving from a research idea or discovery lead to a structured Opportunity Brief, refining it in a workspace, reviewing evidence gaps, and exporting the result.

The broader 2.0 upgrade is a productization roadmap across four areas: AI output quality governance, evaluation automation, model/cost governance, and context/memory safety. The current active work is the AI quality layer: turning recurring AI risks, such as unsupported claims, overconfident recommendations, context loss, and overly broad brief updates, into fixed evaluation scenarios that can be rerun as the product evolves.

This direction connects quality, cost, and context consistency as shared iteration metrics. Human review remains the decision checkpoint for high-risk outputs and external-facing briefs.

## Product Flow

1. The user enters a research project, technology idea, or discovery lead.
2. Specialist agents analyse the opportunity across research, market, timing, and feasibility dimensions.
3. A synthesis agent consolidates the analysis into a structured Opportunity Brief.
4. The user asks follow-up questions in the workspace and decides which outputs should update the brief.
5. A pre-export quality check highlights remaining evidence gaps and decision caveats before PDF export.

## Documentation

- [Product Demonstration](docs/Product-Demonstration.md)
- [Architecture Overview](docs/Architecture-Overview.md)

## Privacy And IP Note

This is a public portfolio showcase. Source code, internal prompts, detailed orchestration logic, deployment configuration, and private assessment materials are not included to protect project IP, deployment security, and assessment integrity. A code walkthrough can be provided during interviews where appropriate.
