# Agent: Retrieval-Augmented Generation (RAG) System

This project implements a modular Retrieval-Augmented Generation (RAG) agent, designed to enhance LLM-based applications with efficient information retrieval and multimodal capabilities. The system is organized for extensibility, maintainability, and ease of use.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Modules Overview](#modules-overview)
- [Extending the System](#extending-the-system)
- [License](#license)

---

## Features

- **Retrieval-Augmented Generation:** Integrates retrieval from a FAISS vector database to ground LLM responses.
- **Image Search & Chart Generation:** Tools for searching images and generating charts.
- **Modular Agent Architecture:** Easily add or modify tools, prompts, and retrieval strategies.
- **Streamlit UI:** Simple web interface for interaction.
- **Configurable Settings:** Centralized configuration for easy environment management.

---

## Project Structure

```
Agent/
├── app.py                  # Main entry point
├── readme.md               # Project documentation
├── agent/
│   ├── image_agent.py      # Image agent logic
│   ├── prompts.py          # Prompt templates
│   ├── runner.py           # Agent runner
│   └── __pycache__/
├── config/
│   ├── settings.py         # Configuration settings
│   └── __pycache__/
├── db/
│   └── db_index.faiss      # FAISS vector index
├── images/                 # Image assets
├── retrieval/
│   ├── search.py           # Retrieval logic
│   └── __pycache__/
├── tools/
│   ├── chart_tool.py       # Chart generation tool
│   ├── image_search_tool.py# Image search tool
│   └── __pycache__/
└── ui/
    └── app.py              # Streamlit UI
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or prepare your FAISS index and place it in `db/db_index.faiss`.**

---

## Configuration

- All configuration settings (API keys, model paths, etc.) are managed in `config/settings.py`.
- Update this file as needed for your environment.

---

## Usage

### 1. **Command Line**

Run the main application:
```bash
python app.py
```

### 2. **Web UI**

Start the Streamlit interface:
```bash
streamlit run ui/app.py
```

---

## Modules Overview

### `agent/`
- **image_agent.py:** Handles image-related agent logic.
- **prompts.py:** Stores and manages prompt templates for LLMs.
- **runner.py:** Orchestrates agent execution.

### `config/`
- **settings.py:** Centralized configuration.

### `db/`
- **db_index.faiss:** FAISS vector index for retrieval.

### `retrieval/`
- **search.py:** Implements retrieval logic using FAISS.

### `tools/`
- **chart_tool.py:** Generates charts from data.
- **image_search_tool.py:** Searches for relevant images.

### `ui/`
- **app.py:** Streamlit-based user interface.

---

## Extending the System

- **Add new tools:** Place new tool modules in `tools/` and register them in the agent.
- **Customize prompts:** Edit or add prompt templates in `agent/prompts.py`.
- **Change retrieval logic:** Modify or extend `retrieval/search.py`.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)

---

For questions or contributions, please open an issue or pull request.
