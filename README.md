# 🎓 SyllabiQ — Autonomous Academic & Syllabus Tracker Agent

**Track:** Agents (Autonomous agents that plan, use tools and finish real tasks end to end)

## Overview
SyllabiQ solves coursework overwhelm for university students. Given a raw syllabus (text or PDF) and a target exam date, the autonomous agent:
1. **Calculates Pacing:** Uses `calculate_study_timeline` to allocate study days and a 20% active-recall buffer before exams.
2. **Researches Live Resources:** Autonomously invokes `search_study_resources` per module to fetch real documentation, lecture notes, and tutorials.
3. **Builds an Interactive Tracker:** Outputs a week-by-week checklist with live progress tracking, Markdown export, and `.ics` calendar integration.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
