# Legal Query RAG (LQ-RAG)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official implementation of "Legal Query RAG: A Retrieval-Augmented Generation Framework with Recursive Feedback for Legal Applications"

## Overview

LQ-RAG explicitly incorporates an agent-based iterative refinement mechanism during inference. It first generates an initial response to a user query and then utilizes an evaluation agent to assess its quality based on contextual relevance and factual grounding. If the response does not meet predefined criteria, the evaluation agent provides feedback to the prompt-engineering agent, which modifies the query to improve the next response. This iterative feedback loop continues until the evaluation scores approach optimal values. The system incorporates:

1. Custom evaluation agent
2. Specialized response generation model
3. Prompt engineering agent
4. Fine-tuned legal embedding LLM

<img src="images/Propose_diagram_color.png" width="1280px" height="720px" />

Key features:
- 13% improvement in Hit Rate
- 15% improvement in Mean Reciprocal Rank (MRR)
- 24% performance gain over general LLMs
- 23% improvement in relevance score over naive configurations

## Installation

```bash
git clone https://github.com/yourusername/legal-query-rag.git
cd legal-query-rag
pip install -r requirements.txt
