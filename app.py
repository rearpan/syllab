import json
import os
import re
from datetime import datetime, timedelta
import streamlit as st
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
from pypdf import PdfReader

# ==========================================
# PAGE CONFIG & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="SyllabiQ — Autonomous Academic Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .agent-log {
        background-color: #0F172A;
        color: #38BDF8;
        font-family: 'Courier New', monospace;
        padding: 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 8px;
        border-left: 4px solid #38BDF8;
    }
    .week-card {
        background-color: #1E293B;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# AGENT TOOLS (AUTONOMOUS CAPABILITIES)
# ==========================================
def search_study_resources(query: str) -> str:
    """
    Searches the web for high-quality study notes, documentation, tutorials,
    and lecture videos for a given academic topic.
    """
    try:
        results = []
        with DDGS() as ddgs:
            search_hits = list(ddgs.text(f"{query} notes tutorial documentation", max_results=4))
            for hit in search_hits:
                results.append({
                    "title": hit.get("title", ""),
                    "url": hit.get("href", ""),
                    "snippet": hit.get("body", "")[:180]
                })
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e), "fallback": f"Search manually for {query}"})


def calculate_study_timeline(exam_date_str: str, total_modules: int) -> str:
    """
    Calculates exact days and weeks remaining until the exam date and allocates
    pacing per module including a final revision buffer.
    """
    try:
        today = datetime.now().date()
        exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
        days_left = (exam_date - today).days
        if days_left <= 0:
            days_left = 14  # Default 2-week crash plan if past/invalid date
        
        total_weeks = max(1, days_left // 7)
        revision_days = max(2, int(days_left * 0.2))
        study_days = max(1, days_left - revision_days)
        days_per_module = round(study_days / max(1, total_modules), 1)

        return json.dumps({
            "today": str(today),
            "exam_date": str(exam_date),
            "days_remaining": days_left,
            "recommended_weeks": total_weeks,
            "active_study_days": study_days,
            "final_revision_buffer_days": revision_days,
            "days_per_module": days_per_module
        })
    except Exception as e:
        return json.dumps({"error": str(e), "recommended_weeks": 4})


# ==========================================
# PDF TEXT EXTRACTOR
# ==========================================
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text.append(content)
    return "\n".join(text)


# ==========================================
# AUTONOMOUS AGENT ORCHESTRATOR
# ==========================================
def run_syllabus_agent(api_key: str, syllabus_text: str, exam_date: str, daily_hours: float, status_container):
    client = genai.Client(api_key=api_key)

    system_instruction = """
    You are an Autonomous Academic Planning & Curriculum Agent.
    Your goal is to take a raw university syllabus and an exam date, and autonomously execute a multi-step workflow:
    1. Analyze the syllabus and count the core modules/units.
    2. Call `calculate_study_timeline` using the exam date and module count to determine exact pacing.
    3. For each major module/topic, call `search_study_resources` to gather real, working URLs (e.g., GeeksforGeeks, NPTEL, official docs, W3Schools, GateSmashers, MIT OCW).
    4. Synthesize everything into a strict, valid JSON object matching this exact schema:
    {
      "course_title": "Detected Course Name",
      "total_weeks": 4,
      "strategy_summary": "Brief strategic advice tailored to the timeline",
      "weeks": [
        {
          "week_number": 1,
          "theme": "Module/Topic Focus",
          "hours_allocated": 10,
          "subtopics": ["Subtopic A", "Subtopic B"],
          "daily_tasks": [
            {"day": "Day 1-2", "task": "Specific actionable study task", "type": "Learn"},
            {"day": "Day 3-4", "task": "Practice problems / code implementation", "type": "Practice"},
            {"day": "Day 5", "task": "Active recall & pyq review", "type": "Revise"}
          ],
          "resources": [
            {"title": "Resource Title", "url": "https://...", "source_type": "Documentation/Notes"}
          ]
        }
      ]
    }
    IMPORTANT: Return ONLY raw JSON in your final response (no markdown fences, no commentary). Ensure every URL comes from your tool outputs.
    """

    user_prompt = f"""
    Target Exam Date: {exam_date}
    Available Study Time: {daily_hours} hours/day
    
    Raw Syllabus Content:
    {syllabus_text[:12000]}
    """

    # Maintain chat history for autonomous tool-calling loop
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]
    tools = [search_study_resources, calculate_study_timeline]

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
        temperature=0.2,
    )

    step = 1
    max_turns = 12  # Prevents infinite loops while allowing multiple tool calls

    while step <= max_turns:
        status_container.markdown(
            f"<div class='agent-log'>🤖 <b>Agent Step {step}:</b> Reasoning & planning next tool execution...</div>",
            unsafe_allow_html=True,
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        # Check if the model decided to invoke tools
        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if part.function_call is not None
        ]

        if not function_calls:
            # Agent finished tool execution and returned final structured plan
            status_container.markdown(
                "<div class='agent-log'>✅ <b>Agent Complete:</b> Syllabus parsed, resources verified, and roadmap compiled!</div>",
                unsafe_allow_html=True,
            )
            raw_text = response.text.strip()
            # Clean accidental markdown code blocks if present
            raw_text = re.sub(r"^```json\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            return json.loads(raw_text)

        # Execute every tool requested by the agent
        tool_response_parts = []
        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args)

            status_container.markdown(
                f"<div class='agent-log'>🛠️ <b>Tool Call:</b> <code>{fn_name}({json.dumps(fn_args)})</code></div>",
                unsafe_allow_html=True,
            )

            if fn_name == "search_study_resources":
                output = search_study_resources(**fn_args)
            elif fn_name == "calculate_study_timeline":
                output = calculate_study_timeline(**fn_args)
            else:
                output = json.dumps({"error": f"Unknown tool {fn_name}"})

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=fn_name,
                    response={"result": output},
                )
            )

        contents.append(types.Content(role="user", parts=tool_response_parts))
        step += 1

    raise RuntimeError("Agent exceeded maximum reasoning steps.")


# ==========================================
# EXPORT HELPERS (MARKDOWN & ICS CALENDAR)
# ==========================================
def generate_markdown_export(plan: dict) -> str:
    md = [f"# 🎓 Study Roadmap: {plan.get('course_title', 'Course')}\n"]
    md.append(f"**Strategy:** {plan.get('strategy_summary', '')}\n---\n")
    for w in plan.get("weeks", []):
        md.append(f"## Week {w['week_number']}: {w['theme']} ({w.get('hours_allocated', 0)} hrs)")
        md.append("**Subtopics:** " + ", ".join(w.get("subtopics", [])))
        md.append("\n### Action Checklist")
        for t in w.get("daily_tasks", []):
            md.append(f"- [ ] **{t['day']}** [{t['type']}]: {t['task']}")
        md.append("\n### Curated Study Resources")
        for r in w.get("resources", []):
            md.append(f"- [{r['title']}]({r['url']}) ({r.get('source_type', 'Web')})")
        md.append("\n---\n")
    return "\n".join(md)


def generate_ics_calendar(plan: dict) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//SyllabiQ Academic Agent//EN",
    ]
    start_date = datetime.now().date()
    for idx, w in enumerate(plan.get("weeks", [])):
        event_start = start_date + timedelta(days=idx * 7)
        event_end = event_start + timedelta(days=6)
        summary = f"Week {w['week_number']}: {w['theme']}"
        desc = "\\n".join([f"- {t['day']}: {t['task']}" for t in w.get("daily_tasks", [])])
        lines.extend([
            "BEGIN:VEVENT",
            f"DTSTART;VALUE=DATE:{event_start.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{event_end.strftime('%Y%m%d')}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{desc}",
            "END:VEVENT",
        ])
    lines.append("END:VCALENDAR")
    return "\n".join(lines)


# ==========================================
# MAIN STREAMLIT UI
# ==========================================
def main():
    st.markdown('<div class="main-header">🎓 SyllabiQ: Autonomous Academic Planner Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Drop your university syllabus & exam date. The agent autonomously splits modules, searches real documentation/notes, and builds an interactive revision tracker.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Agent Configuration")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Get a free API key from Google AI Studio (aistudio.google.com)",
        )

        st.divider()
        st.subheader("📅 Exam Constraints")
        default_exam = datetime.now().date() + timedelta(days=28)
        exam_date = st.date_input("Target Exam Date", value=default_exam, min_value=datetime.now().date())
        daily_hours = st.slider("Daily Study Capacity (Hours)", min_value=1.0, max_value=10.0, value=2.5, step=0.5)

        st.divider()
        st.info(
            "**Agent Architecture:**\n"
            "- **Planner:** Gemini 2.5 Flash\n"
            "- **Tool 1:** `calculate_study_timeline`\n"
            "- **Tool 2:** `search_study_resources` (Live Web Search)\n"
            "- **Output:** Structured JSON + `.ics` Calendar"
        )

    # Sample Syllabus Preset for Instant Demo
    sample_makaut_syllabus = """
    Subject: Data Structures & Algorithms (PCC-CS301) - MAKAUT B.Tech CSE / IoT
    Module 1: Introduction & Asymptotic Analysis (6L)
    Time and space complexity, Big-O, Omega, Theta notations, Recurrence relations, Master Theorem.
    Module 2: Linear Data Structures (10L)
    Arrays, Linked Lists (Singly, Doubly, Circular), Stacks, Infix to Postfix conversion, Queues, Circular Queue, Priority Queue.
    Module 3: Trees & Graph Algorithms (12L)
    Binary Search Trees (BST), AVL Trees, B-Trees, Graph representations (Adjacency Matrix/List), BFS, DFS, Dijkstra's Shortest Path, Kruskal's & Prim's MST.
    Module 4: Sorting & Hashing (8L)
    Quick Sort, Merge Sort, Heap Sort, Radix Sort, Hash functions, Collision resolution (Chaining, Open Addressing).
    """

    col1, col2 = st.columns([1.1, 1.4], gap="large")

    with col1:
        st.subheader("1️⃣ Input Syllabus")
        input_method = st.radio("Choose Input Mode:", ["Paste Syllabus Text", "Upload Syllabus PDF", "Load Sample (MAKAUT CSE/IoT)"], horizontal=True)

        syllabus_text = ""
        if input_method == "Paste Syllabus Text":
            syllabus_text = st.text_area("Paste your syllabus modules here:", height=260, placeholder="Module 1: ...")
        elif input_method == "Upload Syllabus PDF":
            uploaded_pdf = st.file_uploader("Upload University Syllabus PDF", type=["pdf"])
            if uploaded_pdf:
                syllabus_text = extract_text_from_pdf(uploaded_pdf)
                st.success(f"Extracted {len(syllabus_text)} characters from PDF.")
                with st.expander("Preview Extracted Text"):
                    st.text(syllabus_text[:1000] + "...")
        else:
            syllabus_text = st.text_area("Sample Syllabus Loaded:", value=sample_makaut_syllabus, height=260)

        run_btn = st.button("🚀 Launch Autonomous Agent", type="primary", use_container_width=True)

    with col2:
        st.subheader("2️⃣ Agent Live Execution & Output")
        status_box = st.empty()

        if run_btn:
            if not api_key:
                st.error("⚠️ Please enter your Gemini API Key in the left sidebar first.")
            elif not syllabus_text.strip():
                st.warning("⚠️ Please provide syllabus text or upload a PDF.")
            else:
                try:
                    with st.spinner("Agent is analyzing modules and searching the web for notes..."):
                        plan = run_syllabus_agent(
                            api_key=api_key,
                            syllabus_text=syllabus_text,
                            exam_date=str(exam_date),
                            daily_hours=daily_hours,
                            status_container=status_box,
                        )
                        st.session_state["study_plan"] = plan
                        st.session_state["completed_tasks"] = set()
                except Exception as e:
                    st.error(f"Agent execution failed: {e}")

        # Render Generated Study Plan
        if "study_plan" in st.session_state:
            plan = st.session_state["study_plan"]
            st.success(f"### 📚 {plan.get('course_title', 'Generated Study Plan')}")
            st.write(f"**Agent Strategy:** {plan.get('strategy_summary', '')}")

            # Progress Bar Tracker
            all_tasks = []
            for w in plan.get("weeks", []):
                for idx, t in enumerate(w.get("daily_tasks", [])):
                    all_tasks.append(f"w{w['week_number']}_t{idx}")

            total_count = max(1, len(all_tasks))
            done_count = len(st.session_state.get("completed_tasks", set()))
            progress_pct = int((done_count / total_count) * 100)

            st.progress(progress_pct / 100, text=f"Revision Progress: {done_count}/{total_count} tasks completed ({progress_pct}%)")

            # Export Buttons
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    "📥 Download Checklist (.md)",
                    data=generate_markdown_export(plan),
                    file_name="study_roadmap.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with exp_col2:
                st.download_button(
                    "📅 Export to Google/Apple Calendar (.ics)",
                    data=generate_ics_calendar(plan),
                    file_name="exam_schedule.ics",
                    mime="text/calendar",
                    use_container_width=True,
                )

            st.divider()

            # Week-by-Week Interactive Cards
            for w in plan.get("weeks", []):
                with st.expander(f"🗓️ Week {w['week_number']}: {w['theme']} ({w.get('hours_allocated', 0)} hrs)", expanded=(w["week_number"] == 1)):
                    st.markdown("**Core Subtopics:** " + " • ".join([f"`{s}`" for s in w.get("subtopics", [])]))

                    st.markdown("#### ✅ Actionable Revision Checklist")
                    for idx, task in enumerate(w.get("daily_tasks", [])):
                        task_id = f"w{w['week_number']}_t{idx}"
                        checked = st.checkbox(
                            f"**{task['day']}** [{task['type']}]: {task['task']}",
                            key=task_id,
                            value=(task_id in st.session_state["completed_tasks"]),
                        )
                        if checked:
                            st.session_state["completed_tasks"].add(task_id)
                        else:
                            st.session_state["completed_tasks"].discard(task_id)

                    st.markdown("#### 🔗 Agent-Verified Study Resources")
                    resources = w.get("resources", [])
                    if resources:
                        for res in resources:
                            st.markdown(f"- 📄 **[{res['title']}]({res['url']})** — *{res.get('source_type', 'Web Resource')}*")
                    else:
                        st.caption("No external links returned for this module.")


if __name__ == "__main__":
    main()
