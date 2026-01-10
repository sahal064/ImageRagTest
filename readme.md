# Agent: Retrieval-Augmented Generation (RAG) System

This project implements a modular Retrieval-Augmented Generation (RAG) agent, designed to enhance LLM-based applications with efficient information retrieval, image processing, and interactive UI capabilities.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Modules Overview](#modules-overview)
- [Extending the Agent](#extending-the-agent)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Retrieval-Augmented Generation**: Combines LLMs with vector-based retrieval for context-aware responses.
- **Image Processing**: Supports image-based queries and chart generation.
- **Modular Tools**: Easily extendable with custom tools for new functionalities.
- **Interactive UI**: Simple user interface for seamless interaction.
- **Configurable**: Centralized settings for easy customization.

---

## Project Structure

```
Agent/
│
├── app.py                  # Main entry point
├── readme.md               # Project documentation
│
├── agent/                  # Core agent logic
│   ├── image_agent.py
│   ├── prompts.py
│   ├── runner.py
│   └── __pycache__/
│
├── config/                 # Configuration files
│   ├── settings.py
│   └── __pycache__/
│
├── db/                     # Vector database files
│   └── db_index.faiss
│
├── images/                 # Image assets
│
├── retrieval/              # Retrieval logic
│   ├── search.py
│   └── __pycache__/
│
├── tools/                  # Tool modules
│   ├── chart_tool.py
│   ├── image_search_tool.py
│   └── __pycache__/
│
└── ui/                     # User interface
    └── app.py
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Agent
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

- Edit `config/settings.py` to set API keys, model parameters, database paths, and other settings.
- Ensure `db/db_index.faiss` exists or generate it using your data.

---

## Usage

### 1. **Run the Main Application**

```bash
python app.py
```

### 2. **Access the UI**

- Launch the UI with:
  ```bash
  python ui/app.py
  ```
- Open your browser at the indicated address.

---

## Modules Overview

### agent/

- **image_agent.py**: Handles image-based queries and processing.
- **prompts.py**: Stores and manages prompt templates for LLMs.
- **runner.py**: Orchestrates agent workflow and tool invocation.

### config/

- **settings.py**: Central configuration for the agent.

### db/

- **db_index.faiss**: FAISS vector index for fast retrieval.

### retrieval/

- **search.py**: Implements vector search and retrieval logic.

### tools/

- **chart_tool.py**: Generates charts from data.
- **image_search_tool.py**: Finds relevant images for queries.

### ui/

- **app.py**: User interface logic.

---

## Extending the Agent

- **Add New Tools**: Place new tool modules in `tools/` and register them in `agent/runner.py`.
- **Custom Prompts**: Edit or add prompt templates in `agent/prompts.py`.
- **Retrieval Backends**: Swap or extend retrieval logic in `retrieval/search.py`.

---

## Troubleshooting

- **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed.
- **Database Errors**: Verify `db_index.faiss` exists and is accessible.
- **Configuration Issues**: Double-check `config/settings.py` for correct paths and keys.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**For questions or contributions, please open an issue or pull request.**