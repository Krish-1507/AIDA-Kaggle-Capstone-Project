# AIDA: AI Data Automation
**The Enterprise-Grade Autonomous Data Science Agent**

**AIDA** is a cutting-edge **Multi-Agent System (MAS)** designed to democratize advanced data analytics. Built for the **Kaggle AI Agents Intensive Capstone**, it leverages a "Council of Agents" architecture to perform end-to-end data science tasks—from ingestion and cleaning to complex modeling and strategic decision-making—without requiring a single line of code.

---

## Motivation
Data science is powerful but inaccessible. Bridging the gap between raw data and strategic decision-making typically requires a team of data engineers, analysts, and domain experts. Small businesses and non-technical stakeholders often struggle to extract value from their data due to this technical barrier.

**AIDA was built to solve this.** By moving beyond simple chatbots to a fully autonomous **Agentic Workflow**, AIDA acts as an intelligent teammate that doesn't just answer questions but actively investigates data, tests hypotheses, and debates strategies.

---

## Problem Statement
Traditional analytics tools have three major flaws:
1.  **High Technical Barrier:** Requires Python/SQL knowledge.
2.  **Static Analysis:** Dashboards show *what* happened, not *why* or *what to do next*.
3.  **Hallucination Risk:** Standard LLMs often invent data or write broken code when analyzing datasets.

**AIDA's Solution:** A self-healing, multi-agent system that grounds every insight in statistical rigor and executable code.

---

## Architecture
AIDA is built on a modular **Multi-Agent System (MAS)** architecture, where specialized agents collaborate to solve complex problems.

### **The Agentic Workflow**
1.  **Router Agent:** Analyzes the user request and dispatches it to the appropriate specialist (e.g., Hypothesis Engine vs. Causal Agent).
2.  **Worker Agents:**
    *   **Hypothesis Agent:** Scans data for patterns and runs statistical tests (ANOVA, Pearson, Chi-Square).
    *   **Council of Agents:** A multi-persona system (Analyst, Skeptic, CEO) that debates business strategy.
    *   **Causal Agent:** Distinguishes correlation from causation.
3.  **Healer Agent:** A supervisor that monitors code execution. If a script fails, it reads the traceback, debugs the code, and retries automatically.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | Gradio | Custom Glassmorphism UI |
| **LLM Engine** | Google Gemini | Logic & Reasoning (Gemini 1.5 Pro/Flash) |
| **Orchestrator** | Python | Agent Routing & State Management |
| **Data Engine** | Pandas/NumPy | High-performance processing |
| **Vis Engine** | Plotly | Interactive charts |

---

## Key Features

### **Advanced Agentic Intelligence**
*   **Council of Agents:** A unique multi-persona debate system where an **Analyst (Optimist)**, **Risk Officer (Critic)**, and **CEO (Synthesizer)** debate your business questions to reach a balanced strategic decision.
*   **Self-Healing Code Execution:** AIDA doesn't just write code; it fixes it. If a generated Python script fails, the **Healer Agent** analyzes the traceback, debugs the code, and retries automatically.
*   **Hypothesis Engine:** Automatically scans your dataset to discover patterns, generates scientific hypotheses, and validates them using statistical tests.

### **The Ultimate Pipeline**
*   **Universal Ingestion:** Drag-and-drop support for CSV, Excel, JSON, and Parquet.
*   **One-Click Auto-ML:** Automatically trains Random Forest models for Classification and Regression.
*   **Automated Cleaning:** Intelligent imputation of missing values and outlier removal.
*   **Rich Export Bundles:** Generates a deployable ZIP package containing cleaned data, trained models (`.pkl`), and a professional Markdown report.

---

## Experiments
To validate AIDA's capabilities, we tested it on a **Bank Customer Churn Dataset** (`Test_data.csv`).

*   **Dataset Size:** 10,000 rows, 12 features.
*   **Goal:** Understand why customers are leaving and predict future churn.
*   **Process:**
    1.  **Ingestion:** Uploaded raw CSV.
    2.  **Hypothesis Testing:** AIDA automatically hypothesized that `Age` is a significant factor in churn.
    3.  **Validation:** The agent ran a Pearson correlation test and confirmed the hypothesis (p < 0.05).
    4.  **Modeling:** Trained a Random Forest Classifier.

---

## Results
AIDA successfully automated the entire workflow in **under 3 minutes**:

*   **Data Cleaning:** Automatically removed 0 duplicates and handled missing values.
*   **Discovery:** Identified a strong linear relationship between **Age** and **Exited** status.
*   **Model Performance:** The Random Forest model achieved an accuracy of **86%** on the validation set.
*   **Strategic Decision:** The Council of Agents debated a "Loyalty Program for Seniors" and concluded it was a high-ROI move based on the data evidence.

---

## Demo
Watch AIDA in action:
**[Link to Video Demo]** *(Insert your video link here)*

The demo showcases:
1.  **Real-time Data Ingestion & Cleaning.**
2.  **The "Council of Agents" debating a live business strategy.**
3.  **The "Hypothesis Engine" discovering hidden patterns.**
4.  **Rich Export of the entire session.**

---

## Installation

### **Prerequisites**
*   Python 3.9+
*   A valid [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### **Setup**
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/aida-capstone.git
    cd aida-capstone
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python main.py
    ```
    Access the platform at `http://127.0.0.1:7870`.

---

## Conclusion
AIDA demonstrates the transformative potential of **Agentic AI**. By combining the reasoning capabilities of LLMs with the rigor of statistical tools and self-healing code execution, it bridges the gap between raw data and actionable strategy. It is not just a tool; it is an intelligent teammate for the modern enterprise.

---

*Built by Krish J for the Kaggle AI Agents Intensive Capstone.*
