Search Backend for Wikipedia Corpus

Overview

This search backend is designed to efficiently index and query the English Wikipedia corpus. Utilizing a combination of BM25, PageRank, and other optimizations, our system aims to deliver precise and relevant search results.

Features

BM25 Indexing: Calculates partial BM25 scores for document-term relevance.
PageRank Scores: Incorporates PageRank for document importance.
Multi-threading: Enhances search performance through concurrent processing.
Flexible Query Handling: Supports complex queries with optimized tokenization and stemming.

Code Structure and Functionality

The codebase is organized into several key components, each responsible for a distinct aspect of the search engine's functionality:

1. Inverted Index (inverted_index_gcp.py)

This module is responsible for creating and managing the inverted index. It stores mappings from terms to the documents that contain them, along with BM25 partial scores for each term-document pair to expedite relevance scoring during query processing.

Functionality: Includes methods for indexing documents, writing and reading index files to and from Google Cloud Storage, and calculating BM25 scores.

2. Query Processing (search_backend.py)

The core of the search engine, this script processes search queries by interacting with the inverted index and applying relevance scoring to retrieve and rank documents.

Key Components:

Tokenization and Stemming: Breaks down queries into tokens and applies stemming to reduce words to their root form, improving the matching process.
BM25 Scoring: Utilizes pre-computed partial scores from the index, combining them with query-specific calculations to determine document relevance.
PageRank Integration: Retrieves PageRank scores for documents from a CSV file and integrates these into the final ranking to prioritize important documents.
Multithreading Optimization: Implements multithreading in querying to parallelize the retrieval and scoring of documents, significantly reducing response times.

3. Data Management (data_utils.py)

Though not explicitly shown in the initial code outline, this hypothetical module would be responsible for managing interactions with data sources, such as downloading and parsing Wikipedia dumps and interfacing with Google Cloud Storage for index storage and retrieval.

Functionality: Includes functions for parsing Wikipedia data, uploading and downloading index files, and converting data into formats suitable for indexing and querying.
4. Evaluation and Analysis (evaluation.py)
Another suggested module for evaluating the search engine's performance, including calculating precision, recall, and query processing times, and generating performance graphs.

Functionality: Provides tools for running benchmark queries against the search engine, comparing results against a ground truth set, and visualizing performance over time or across different configurations.
