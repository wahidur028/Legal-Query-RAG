# 🏛️ Legal Query RAG (LQ-RAG)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> Official implementation of **"Legal Query RAG: A Retrieval-Augmented Generation Framework with Recursive Feedback for Legal Applications"**  
> Published in *IEEE Access* · [Read the paper ↗](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10887211)

### 👥 Authors
- [**Rahman S M Wahidur**](https://scholar.google.com/citations?user=0_GwJz4AAAAJ&hl=en&oi=ao), [**SUMIN KIM**], [**HAEUNG CHO**], [**DAVID S. BHATTI**](https://scholar.google.com/citations?user=RU0j8cMAAAAJ&hl=en), [**Heung-No Lee**](https://scholar.google.com/citations?user=lRlN_40AAAAJ&hl=en)
---

## 🔍 Overview

LQ-RAG explicitly incorporates an agent-based iterative refinement mechanism during inference. It first generates an initial response to a user query and then utilizes an evaluation agent to assess its quality based on contextual relevance and factual grounding. If the response does not meet predefined criteria, the evaluation agent provides feedback to the prompt-engineering agent, which modifies the query to improve the next response. This iterative feedback loop continues until the evaluation scores approach optimal values.

---

## 🧠 Architecture Components

- **Custom Evaluation Agent** – custom scorer to evaluate factual correctness and legal context
- **Fine-Tuned Response Generator** – LLM fine-tuned on legal texts
- **Prompt Engineering Agent** – dynamically adapts queries based on evaluation feedback
- **Legal Embedding Model** – specialized vector store for legal document retrieval

<p align="center">
  <img src="img/execution flow.png"/>
</p>
---

## 📈 Key Results

- **+13%** Hit Rate improvement
- **+15%** boost in Mean Reciprocal Rank (MRR)
- **+24%** performance gain over general LLM baselines
- **+23%** relevance score improvement vs. naive RAG setups
---

## 🚀 Installation

Follow these steps to install and run the project:

```bash
# Clone the repository
git clone https://github.com/wahidur028/Legal-Query-RAG.git
cd Legal-Query-RAG

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

## 📈 Key Results

- **+13%** Hit Rate improvement
- **+15%** boost in Mean Reciprocal Rank (MRR)
- **+24%** performance gain over general LLM baselines
- **+23%** relevance score improvement vs. naive RAG setups
---
