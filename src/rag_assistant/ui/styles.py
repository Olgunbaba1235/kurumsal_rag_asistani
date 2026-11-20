"""Özel CSS — Streamlit temasının üzerine ince bir kurumsal katman."""

CUSTOM_CSS = """
<style>
:root {
  --rag-accent: #4F46E5;
  --rag-accent-soft: rgba(79, 70, 229, 0.10);
  --rag-border: rgba(128, 128, 128, 0.22);
}

.block-container { padding-top: 2rem; max-width: 980px; }

.rag-hero {
  padding: 1.4rem 1.6rem;
  border-radius: 16px;
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #0EA5E9 100%);
  color: #fff;
  margin-bottom: 1.2rem;
  box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.rag-hero h1 { color: #fff; font-size: 1.7rem; margin: 0 0 .25rem 0; padding: 0; }
.rag-hero p  { margin: 0; opacity: .92; font-size: .95rem; }

.rag-stats { display: flex; gap: .75rem; margin: .4rem 0 1rem 0; flex-wrap: wrap; }
.rag-stat {
  flex: 1 1 120px; padding: .7rem .9rem; border-radius: 12px;
  border: 1px solid var(--rag-border); background: var(--rag-accent-soft);
}
.rag-stat .v { font-size: 1.35rem; font-weight: 700; }
.rag-stat .l { font-size: .75rem; opacity: .75; text-transform: uppercase; letter-spacing: .04em; }

.rag-source {
  border: 1px solid var(--rag-border); border-left: 4px solid var(--rag-accent);
  border-radius: 10px; padding: .65rem .85rem; margin-bottom: .6rem;
}
.rag-source.cited { border-left-color: #10B981; }
.rag-source .hdr { display: flex; justify-content: space-between; gap: .5rem; font-size: .85rem; font-weight: 600; }
.rag-source .score { font-variant-numeric: tabular-nums; opacity: .75; font-weight: 500; }
.rag-source .body { font-size: .83rem; opacity: .85; margin-top: .35rem; white-space: pre-wrap; }

.rag-badge {
  display: inline-block; font-size: .68rem; padding: .05rem .45rem; border-radius: 999px;
  background: var(--rag-accent-soft); border: 1px solid var(--rag-border); margin-left: .35rem;
}

.rag-doc-row { font-size: .88rem; }
</style>
"""
