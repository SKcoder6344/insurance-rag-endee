"""
PolicyGuard AI — Streamlit Chat Interface

Visual demo of the 3-step agentic pipeline:
  Step 1 → Coverage search
  Step 2 → Exclusions + waiting periods
  Step 3 → Claim procedure

Each step is shown live with retrieved chunks and similarity scores.
Final verdict is displayed with colour-coded badge.
"""
import time
import os
import requests
import streamlit as st

# Reads from env var on Streamlit Cloud, falls back to localhost for local dev
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolicyGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Verdict badges */
.badge-covered {
    background: #d4edda; border-left: 5px solid #28a745;
    padding: 1rem 1.25rem; border-radius: 8px; margin: 0.5rem 0;
}
.badge-not-covered {
    background: #f8d7da; border-left: 5px solid #dc3545;
    padding: 1rem 1.25rem; border-radius: 8px; margin: 0.5rem 0;
}
.badge-partial {
    background: #fff3cd; border-left: 5px solid #ffc107;
    padding: 1rem 1.25rem; border-radius: 8px; margin: 0.5rem 0;
}
/* Agent step cards */
.step-card {
    background: #f8f9fa; border: 1px solid #dee2e6;
    border-radius: 8px; padding: 0.85rem 1rem; margin: 0.4rem 0;
}
.step-number {
    font-weight: 700; color: #4361ee; font-size: 0.9rem;
}
/* Similarity pill */
.sim-pill {
    background: #e9ecef; color: #495057;
    padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
    font-family: monospace;
}
/* Section tag */
.section-tag {
    background: #cfe2ff; color: #084298;
    padding: 2px 7px; border-radius: 10px; font-size: 0.75rem;
}
/* Chat bubbles */
.user-bubble {
    background: #4361ee; color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1rem; margin: 0.25rem 0;
    max-width: 75%; float: right; clear: both;
}
.assistant-bubble {
    background: #f0f2f5;
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1rem; margin: 0.25rem 0;
    max-width: 85%; float: left; clear: both;
}
.clearfix { clear: both; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/PolicyGuard-AI-4361ee?style=for-the-badge&logo=shield")
    st.markdown("### 🛡️ PolicyGuard AI")
    st.markdown(
        "**Agentic Insurance Claims Analyzer** powered by "
        "[Endee](https://endee.io) Hybrid Vector Search + Groq LLaMA3."
    )
    st.divider()

    st.markdown("#### 🔬 How it works")
    st.markdown("""
1. **Step 1** — Hybrid search on *coverage terms*
2. **Step 2** — Hybrid search on *exclusions + waiting periods*
3. **Step 3** — Hybrid search on *claim procedure*
4. **Synthesis** — Groq LLaMA3 generates a structured verdict
    """)
    st.divider()

    st.markdown("#### 🧪 Try these claims")
    sample_claims = [
        "My father has Type 2 diabetes and needs a kidney transplant. Is it covered?",
        "I need cataract surgery in both eyes. My policy is 3 months old.",
        "I was in a road accident and need knee replacement surgery immediately.",
        "My wife is 8 months pregnant. Will the delivery be covered?",
        "I want LASIK surgery for my eyes. Will insurance pay for it?",
        "I had a heart attack and was taken to a non-network hospital by ambulance.",
    ]
    for claim in sample_claims:
        if st.button(f"💬 {claim[:55]}...", use_container_width=True, key=claim):
            st.session_state.prefill = claim

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # API health
    st.divider()
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ API connected")
        else:
            st.error("⚠️ API error")
    except Exception:
        st.error("❌ API offline — run `docker compose up`")


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ PolicyGuard AI")
st.caption(
    "Agentic Insurance Claims Analyzer — Endee Hybrid Search (Dense + BM25) + Groq LLaMA3"
)
st.divider()


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🛡️"):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            _render_verdict(msg["content"])


# ── Helper: render a verdict dict ─────────────────────────────────────────────
def _verdict_badge(verdict: str) -> str:
    icons = {"COVERED": "✅", "NOT_COVERED": "❌", "PARTIAL": "⚠️"}
    css = {
        "COVERED": "badge-covered",
        "NOT_COVERED": "badge-not-covered",
        "PARTIAL": "badge-partial",
    }
    icon = icons.get(verdict, "❓")
    cls = css.get(verdict, "badge-partial")
    label = verdict.replace("_", " ")
    return f'<div class="{cls}"><b>{icon} {label}</b></div>'


def _render_verdict(data: dict) -> None:
    """Render the full verdict response with agent steps."""
    verdict = data.get("verdict", "UNKNOWN")

    # Verdict badge
    st.markdown(_verdict_badge(verdict), unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{data.get('summary', '')}**")
    with col2:
        conf = data.get("confidence", 0)
        st.metric("Confidence", f"{conf:.0%}")

    # Coverage & exclusions
    col_a, col_b = st.columns(2)
    with col_a:
        if data.get("coverage_points"):
            st.markdown("**✅ Coverage applies:**")
            for pt in data["coverage_points"]:
                st.markdown(f"- {pt}")
    with col_b:
        if data.get("exclusion_points"):
            st.markdown("**⛔ Exclusions / Conditions:**")
            for pt in data["exclusion_points"]:
                st.markdown(f"- {pt}")

    wp = data.get("waiting_period", "None")
    if wp and wp.lower() != "none":
        st.warning(f"⏳ **Waiting Period:** {wp}")

    # Claim steps
    if data.get("claim_steps"):
        with st.expander("📋 How to file this claim"):
            for step in data["claim_steps"]:
                st.markdown(f"- {step}")

    if data.get("recommendation"):
        st.info(f"💡 **Recommendation:** {data['recommendation']}")

    # Agent steps (retrieval trace)
    with st.expander("🔍 Agent reasoning trace (Endee hybrid search steps)"):
        steps = data.get("steps", [])
        for step in steps:
            step_icons = {1: "🔎", 2: "⛔", 3: "📋"}
            icon = step_icons.get(step.get("step"), "•")
            st.markdown(
                f'<div class="step-card"><span class="step-number">'
                f'{icon} Step {step["step"]}: {step["action"]}</span><br>'
                f'<small>Query: <i>{step["query"]}</i></small></div>',
                unsafe_allow_html=True,
            )
            for chunk in step.get("results", [])[:3]:
                sim = chunk.get("similarity", 0)
                section = chunk.get("section", "")
                text = chunk.get("text", "")[:180]
                st.markdown(
                    f'<div style="margin-left:1rem;margin-top:4px;">'
                    f'<span class="section-tag">{section}</span> '
                    f'<span class="sim-pill">sim={sim:.3f}</span> '
                    f'<span style="font-size:0.85rem;color:#555;">{text}…</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    latency = data.get("latency_ms")
    if latency:
        st.caption(f"⚡ Total latency: {latency:.0f}ms")


# ── Input ─────────────────────────────────────────────────────────────────────
prefill_val = st.session_state.get("prefill", "")
if prefill_val:
    st.session_state.prefill = ""  # clear after use

claim_input = st.chat_input(
    placeholder="Describe your insurance claim… e.g. 'My father has diabetes and needs knee surgery'",
    key="claim_input",
)

# Allow sidebar sample claims to prefill
if prefill_val and not claim_input:
    claim_input = prefill_val

# ── On submit ─────────────────────────────────────────────────────────────────
if claim_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": claim_input})
    with st.chat_message("user", avatar="🧑"):
        st.write(claim_input)

    # Run agent
    with st.chat_message("assistant", avatar="🛡️"):
        with st.status("🤖 Agent working…", expanded=True) as status:
            st.write("🔎 Step 1: Searching coverage terms in Endee…")
            time.sleep(0.3)
            st.write("⛔ Step 2: Checking exclusions & waiting periods…")
            time.sleep(0.3)
            st.write("📋 Step 3: Retrieving claim procedure…")
            time.sleep(0.2)
            st.write("🧠 Synthesising verdict with Groq LLaMA3…")

            try:
                resp = requests.post(
                    f"{API_URL}/analyze",
                    json={"claim": claim_input},
                    timeout=60,
                )
                if resp.status_code == 200:
                    verdict_data = resp.json()
                    status.update(label="✅ Analysis complete", state="complete")
                    _render_verdict(verdict_data)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": verdict_data}
                    )
                else:
                    status.update(label="❌ Error", state="error")
                    st.error(f"API error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                status.update(label="❌ Connection failed", state="error")
                st.error(
                    "Cannot reach the API. Make sure you ran:\n```\ndocker compose up -d\n```"
                )
            except requests.exceptions.Timeout:
                status.update(label="❌ Timeout", state="error")
                st.error("Request timed out. The agent may be loading models for the first time.")
