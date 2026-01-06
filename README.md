# RAG-Based Image Retrieval & Querying(Ongoing)

## Problem Statement

The goal of this project is to:

- Retrieve images from storage using natural language text queries
- Generate captions for images to enrich semantic understanding
- Enable querying and inference over image data rather than simple retrieval

---

## Challenges

- Simple images do not contain any implicit structure that can be directly used for mathematical or statistical reasoning.
- Images lack explicit semantic meaning, and visual similarity cannot be directly equated to semantic similarity.
- Abstract user queries (for example, *How much?*, *Which?*, *What entity?*) cannot be answered using basic image retrieval.
- Images containing data such as charts, plots, and documents encode information visually, making them difficult to ingest or query directly.

---

## Limitations of Current Systems

- They can generate embeddings but cannot provide explanations.
- They lack the ability to understand the structured meaning of visual data.
- There is no grounding or justification for why a particular image was retrieved.

---

## Goal

To build a RAG-based system that:

- Focuses on retrieval using embeddings, combined with reasoning over retrieved evidence
- Enables grounded and explainable responses
- Supports multimodal reasoning using:
  - Visual embeddings
  - Textual evidence
  - Structured metadata
- Adapts to user intent over time through continuous learning

---

## End-to-End Pipeline

- **User Input**  
  Users submit queries as natural language text.

- **Embedding-Based Retrieval**  
  The query is converted into a vector embedding and used to retrieve the most visually and semantically similar images from a vector database.

- **Metadata-Based Filtering**  
  Tags and captions are used to pre-filter and refine the candidate image set, improving speed and relevance.

- **Image Routing**  
  Each retrieved image is passed through a routing layer to determine whether it is:
  - A natural image  
  - A structured visual such as a chart or graph

- **Conditional Processing**
  - Natural images are processed using visual similarity and caption-based reasoning.
  - Charts and graphs are processed using OCR and structured interpretation to extract meaningful evidence.

- **Structured Representation**  
  Extracted evidence is converted into structured intermediate representations that capture only observable and reliable information.

- **Grounded Generation**  
  These representations are passed to a language model that generates a final response strictly grounded in retrieved evidence.

---

## Major Components

- **Image Encoder**  
  Converts images into semantic embeddings for similarity-based retrieval.

- **Text Encoder**  
  Transforms text queries into embeddings compatible with image embeddings.

- **Vector Database (FAISS)**  
  Stores image embeddings and enables fast similarity search at query time.

- **Metadata Store**  
  Maintains captions, tags, and image paths to support filtering, ranking, and explainability.

- **Captioning Module**  
  Generates descriptive captions offline to add semantic context for retrieval and reasoning.

- **Routing & Image Classifier**  
  Identifies whether an image is a natural photo or a structured visual and routes it to the correct pipeline.

- **Chart Understanding Pipeline**  
  Interprets statistical charts using OCR and structural analysis to generate safe, high-level insights.

- **Retrieval-Augmented Generation (RAG)**  
  Combines retrieved images, metadata, and structured evidence to produce grounded responses.

- **Learning & Tag Update Mechanism**  
  Continuously enriches metadata by learning from user queries, improving retrieval relevance over time without retraining models.

---

## Limitations

- Chart interpretation is approximate rather than exact.
- OCR inaccuracies may affect text extraction quality.
- Not all chart types, particularly line graphs, are currently handled.
- No domain-specific validation of extracted data.
- Charts with heavy graphics or illustrative formatting may not be processed correctly.

---

## Future Work

- Support for line and pie charts
- Table extraction from images
- Enhanced document understanding
- Dashboard and multi-chart handling
- Confidence scoring for extracted insights
- Incremental indexing for scalable updates
