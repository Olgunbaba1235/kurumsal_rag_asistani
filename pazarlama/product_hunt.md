# Product Hunt Launch Kit

> Launch at 00:01 PT (Product Hunt day starts in Pacific Time). Live demo: https://jxq2ypt97fhsweii9fwk9r.streamlit.app/ · Replace `[WEBSITE_LINK]` once the landing page is hosted.
> Gallery: 1270×760 images — (1) hero with an "Auditor Approved" answer, (2) source cards with page numbers,
> (3) "Not found — fabrication blocked" refusal, (4) department filter explaining a refusal, (5) analytics tab.

## Name
**Kurumsal RAG Asistanı** — Zero-Hallucination Document Assistant

## Tagline (≤60 chars)
**Answers from your company docs — or says "not found"**

Alternatives:
- AI that cites the page — and never makes things up
- Your policies, answered with sources. Zero guessing.

## Topics
Artificial Intelligence · SaaS · Productivity · Legal · Human Resources

## Description (≤260 chars)
An AI assistant for company documents. Every answer cites the exact document and page, a second "Auditor" agent strips any number or claim not found in the sources, and if the answer isn't in your docs it simply says so. Try the live demo — no signup.

## Maker's first comment

Hi Product Hunt 👋

We built Kurumsal RAG Asistanı ("corporate RAG assistant") after seeing the same thing at every mid-sized company: employees keep asking HR and finance the same policy questions, and generic chatbots answer confidently even when they're wrong.

So we designed around one rule: **if it's not in your documents, the assistant doesn't say it.**

How it works:
- 🔎 **Hybrid search** — semantic vectors + BM25 keywords + reranking, with a relevance gate. Below the threshold, the model is never called.
- 📎 **Page-level citations** — document name, page, relevance score and the quoted passage (tables included).
- 🛡️ **Researcher + Auditor agents** — the Auditor checks every claim against the retrieved passages and removes unsupported numbers, dates, rates and quotes. Each answer ships with an Audit Report: Approved / Revised / Rejected.
- 🧩 **Compound questions** — answers the part your docs cover and explicitly notes the part they don't.
- 🏢 **Department-scoped search & multi-tenant isolation** — HR, Finance, Legal… with row-level security per company.

The live demo runs in your browser with sample HR & finance policies (in Turkish). Hit "▶ Start" and watch question #5 get refused instead of hallucinated.

We'd love your feedback — especially from legal, accounting and HR teams: what would you need before trusting an AI with your internal documents?

Demo: https://jxq2ypt97fhsweii9fwk9r.streamlit.app/ · Website: [WEBSITE_LINK]

## Short reply templates
- **"How is this different from ChatGPT with file upload?"** — Retrieval is gated by a relevance threshold, every sentence must carry a citation, and an independent Auditor agent removes unsupported claims before you see the answer. Out-of-scope questions get an explicit "not in your documents".
- **"Where is data stored?"** — Each company gets an isolated workspace (PostgreSQL row-level security). Self-hosted (Docker) deployment is available for teams that need it.
- **"English support?"** — The demo ships with Turkish sample documents and the search is tuned for Turkish. English documents can be indexed; we'd validate answer quality on your files in a pilot first.
