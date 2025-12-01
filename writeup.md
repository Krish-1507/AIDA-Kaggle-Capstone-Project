# AIDA: The Enterprise-Grade Autonomous Data Science Agent

## The Pitch

### Problem Statement
Data science is powerful but inaccessible. Bridging the gap between raw data and strategic decision-making typically requires a team of data engineers, analysts, and domain experts. Small businesses and non-technical stakeholders often struggle to extract value from their data due to this technical barrier.

Traditional analytics tools have three major flaws:
1.  **High Technical Barrier:** Requires Python/SQL knowledge.
2.  **Static Analysis:** Dashboards show *what* happened, not *why* or *what to do next*.
3.  **Hallucination Risk:** Standard LLMs often invent data or write broken code when analyzing datasets.

### Solution: AIDA
**AIDA (AI Data Automation)** is a cutting-edge **Multi-Agent System (MAS)** designed to democratize advanced data analytics. Built for the **Enterprise Agents** track, it leverages a "Council of Agents" architecture to perform end-to-end data science tasks—from ingestion and cleaning to complex modeling and strategic decision-making—without requiring a single line of code.

AIDA acts as an intelligent teammate that doesn't just answer questions but actively investigates data, tests hypotheses, and debates strategies.

### Value Proposition
AIDA reduces the time-to-insight from days to minutes. It empowers non-technical users to perform tasks that previously required a data scientist, such as:
*   **Scientific Hypothesis Testing:** Automatically discovering and validating patterns (e.g., "Age correlates with Churn").
*   **Strategic Planning:** Simulating a boardroom debate to weigh pros and cons of business decisions.
*   **Self-Healing Automation:** Executing Python code that fixes itself if errors occur.

---

## The Implementation

### Architecture
AIDA is built on a modular **Multi-Agent System (MAS)** architecture, where specialized agents collaborate to solve complex problems.

**1. The Agentic Workflow:**
*   **Router Agent:** Analyzes the user request and dispatches it to the appropriate specialist.
*   **Worker Agents:**
    *   **Hypothesis Agent:** Scans data for patterns and runs statistical tests (ANOVA, Pearson, Chi-Square).
    *   **Council of Agents:** A multi-persona system (Analyst, Skeptic, CEO) that debates business strategy.
    *   **Causal Agent:** Distinguishes correlation from causation.
*   **Healer Agent:** A supervisor that monitors code execution. If a script fails, it reads the traceback, debugs the code, and retries automatically.

**2. Tech Stack:**
*   **Frontend:** Gradio (Custom Glassmorphism UI)
*   **LLM Engine:** Google Gemini 1.5 Pro & Flash
*   **Orchestrator:** Python (Custom Agent Routing)
*   **Data Engine:** Pandas/NumPy
*   **Observability:** Custom `AgentLogger` for real-time thought tracking.

### Key Concepts Applied
This project demonstrates mastery of the following course concepts:
1.  **Multi-Agent Systems:** The "Council of Agents" and "Healer Agent" work in concert to solve problems no single agent could handle.
2.  **Tools & Code Execution:** Agents have access to a Python execution environment to run statistical tests and train models.
3.  **Observability:** A dedicated "Agent Logs" system tracks the reasoning steps of every agent in real-time.

---

## Experiments & Results

To validate AIDA's capabilities, we tested it on a **Bank Customer Churn Dataset** (10,000 rows).

**The Workflow:**
1.  **Ingestion:** Uploaded raw CSV.
2.  **Hypothesis Testing:** AIDA automatically hypothesized that `Age` is a significant factor in churn.
3.  **Validation:** The agent ran a Pearson correlation test and confirmed the hypothesis (p < 0.05).
4.  **Modeling:** Trained a Random Forest Classifier with **86% accuracy**.
5.  **Strategy:** The Council of Agents debated a "Loyalty Program for Seniors" and concluded it was a high-ROI move based on the data evidence.

**Result:** The entire process took **under 3 minutes**, compared to hours of manual work.

---

## Demo Video
**[Insert YouTube Link Here]**

The demo showcases:
1.  **Real-time Data Ingestion & Cleaning.**
2.  **The "Council of Agents" debating a live business strategy.**
3.  **The "Hypothesis Engine" discovering hidden patterns.**
4.  **Rich Export of the entire session.**

---

## Conclusion
AIDA demonstrates the transformative potential of **Agentic AI**. By combining the reasoning capabilities of LLMs with the rigor of statistical tools and self-healing code execution, it bridges the gap between raw data and actionable strategy. It is not just a tool; it is an intelligent teammate for the modern enterprise.
