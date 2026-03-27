---
trigger: always_on
---

# Workflow Orchestration

## 0. Core Repo Principle Dissection (Mandatory First Action)

- **Trigger**: Every time the repository is opened OR a new session/task begins.
- **Non-negotiable rule**: Before any planning, research, or coding, dissect and document the single underlying principle/idea that the entire repo exists to embody.

**Exact dissection steps (always executed in order):**
1. Scan README, docs, main entrypoints, architecture diagrams, and key source files to extract the **core idea** in one crisp sentence.
2. Articulate the **exact intuition**: Why does this principle feel magically correct? What deep problem does it solve that every obvious alternative fails at?
3. Ground it rigorously: Link to the scientific/mathematical foundations or state-of-the-art research (include DOIs/links per current rule 0).
4. Enumerate and reject alternatives: List the top 2–3 other viable architectures/ideas and prove (with evidence) why the repo’s chosen principle is superior.
5. Write/update `tasks/core_principle.md` with the manifesto (keep it < 1 page). Title it “This Repo’s North Star”.
6. From this point forward, **every** plan, subagent task, code change, or verification step must explicitly reference this principle and justify alignment. Any drift is flagged immediately and triggers a re-plan.

- If the principle is ambiguous or undocumented: Use subagents + literature review to infer it from behavior and code structure (never guess). If still unclear, note it in the manifesto and ask for one-sentence clarification—then lock it in.
- This step is the new “root node” of the entire workflow. All other rules (research, planning, elegance, verification, self-improvement) are downstream of it.

**Why this is non-negotiable**: Misunderstanding the repo’s core idea is the only error that cannot be fixed by better code or more tests. Capturing the exact intuition here eliminates that class of failure permanently.

## 1. Research-Driven Development (State-of-the-Art Priority)
- For every non-trivial task: begin with scientific literature review and reference research articles (include DOIs/links)
- Deeply understand and articulate the intuition, mathematical foundations, and scientific principles behind the code
- Explicitly enumerate and rigorously evaluate ALL viable options/architectures before suggesting any plan
- Prioritize state-of-the-art approaches backed by recent peer-reviewed research

## 2. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

## 3. Subagent Strategy
- Use subagents liberally to keep main contect window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

## 4. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

## 5. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

## 6. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, chvious fixes – don't over-engineer
- Challenge your own work before presenting it

## 7. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

# Task Management
1. **Plan First**: Write plan to `tasks/todo.md` with checkable items  
2. **Verify Plans**: Check in before starting implementation  
3. **Track Progress**: Mark items complete as you go  
4. **Explain Changes**: High-level summary at each step  
5. **Document Results**: Add review section to `tasks/todo.md`  
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections  

---

# Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.  
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.  
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.