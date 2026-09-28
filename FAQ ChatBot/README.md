# ShopAssist - E-Commerce FAQ Chatbot

ShopAssist is a domain-specific customer-support chatbot that finds a relevant answer from an e-commerce FAQ knowledge base. It uses text preprocessing, TF-IDF vectors, and cosine similarity rather than question-specific conditional responses.

## Problem Statement

E-commerce support teams repeatedly answer questions about orders, delivery, returns, refunds, payment, and account access. Finding the right answer manually takes time, while a generic chatbot may make unsupported claims.

## Objective

Provide a small, runnable support chatbot that matches a customer's wording to a curated FAQ and returns the corresponding answer. When the best match is not strong enough, ShopAssist says it cannot find a reliable answer instead of inventing one.

## Features

- 36 e-commerce FAQ entries across 11 support categories.
- NLTK-based lowercase tokenization and stopword removal.
- Scikit-learn TF-IDF vectorization with unigram and bigram features.
- Cosine similarity ranking and a configurable answer threshold.
- Clear fallback for unrelated or low-similarity questions.
- Streamlit chat interface with suggested questions, conversation history, and a clear-chat control.
- Subtle display of the best cosine similarity for each response.
- Graceful handling of a missing FAQ file and invalid FAQ records.
- Works without downloading NLTK data at runtime; a built-in stopword fallback is used if the NLTK corpus is unavailable.

## How the Chatbot Works

1. `app.py` loads FAQ records from `faq_data.json` and skips malformed entries.
2. FAQ questions are lowercased, tokenized with NLTK, and stripped of English stopwords.
3. A scikit-learn `TfidfVectorizer` transforms the processed FAQ questions into vectors.
4. Each user question follows the same preprocessing and vectorization path.
5. Cosine similarity ranks the FAQ vectors against the user-question vector.
6. The answer for the best match is returned only when its similarity is at least `0.18`. Otherwise the chatbot displays the fallback message.

The similarity shown in chat is a cosine similarity score, not a probability or an AI confidence estimate. The threshold is a simple baseline and should be tuned against representative questions for a production deployment.

## NLP Techniques Used

- **Lowercasing:** normalizes differences in capitalization.
- **Tokenization:** NLTK's regular-expression tokenizer splits text into word and number tokens while excluding punctuation.
- **Stopword removal:** common English function words are removed; the NLTK stopword corpus is used when present, with a local fallback when it is not.
- **TF-IDF:** weights terms by their usefulness across the FAQ question collection.
- **Cosine similarity:** compares the user's question vector with each FAQ vector and ranks the closest match.
- **Similarity threshold:** avoids returning an answer when the best match is too weak.

## Technology Stack

- Python
- Streamlit
- NLTK
- scikit-learn
- JSON

## Project Structure

```text
ShopAssist/
|-- app.py
|-- faq_data.json
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## Installation

Use Python 3.10 or later. From the project directory, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. No API key or external service is required.

## Example Questions

- Where is my package?
- I want to cancel my order.
- Can I get my money back?
- How do I return something?
- What cards can I use?
- My product arrived broken.
- Can I change my delivery address?
- Who won the cricket match?

The first seven should match relevant FAQ entries. The unrelated cricket question should receive the fallback response.

## Limitations

- The chatbot can only answer questions represented by the curated FAQ wording; TF-IDF does not understand intent or context like a language model.
- Matching is lexical, so unusual synonyms and multi-part questions may not find the expected FAQ.
- Each message is matched independently; conversation context, order lookup, account authentication, and live shipment data are not implemented.
- FAQ answers are example store policies and should be reviewed and replaced with the actual retailer's policies before customer use.
- The similarity threshold is a heuristic and should be evaluated on real customer questions.

## Future Enhancements

- Add a labeled evaluation set and tune the threshold using precision and recall.
- Expand normalization with stemming, lemmatization, or domain synonym handling.
- Add multilingual FAQs and language detection.
- Connect authenticated order and shipment lookups through a secure commerce API.
- Collect unanswered questions for support-team review and knowledge-base updates.
- Add automated tests and deployment configuration for a hosted environment.