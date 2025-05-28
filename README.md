
## Installation & Setup

### Requirements

* Python 3.8+
* `pip` package manager
* GPU recommended for model inference (optional but faster)
* Internet connection to install packages and use Replicate API

### Key Python Libraries

```bash
pip install streamlit pandas numpy faiss-cpu sentence-transformers replicate python-dotenv
```

> Note: On some systems, install `faiss-gpu` if GPU support is needed.

---

## Environment Variables

Create a `.env` file in the project root with:

```
REPLICATE_API_TOKEN=your_replicate_api_key_here
```

This token is required for using the Replicate API for LLM inference.
I've commented out API key in the app.py file at the end, use it in .env file 

---


## How to Run

### 1. Data Preparation & Model Fine-Tuning

* Open `modelling.ipynb`
* Follow the notebook to clean and prepare the dataset.
* Generate embeddings with SentenceTransformer.
* Save embeddings and FAISS index.


### 2. Streamlit Application

Run the app with:

```bash
streamlit run app.py
```

* Enter queries such as:

  * “Quotes about insanity attributed to Einstein”
  * “Motivational quotes tagged ‘accomplishment’”
  * “All Oscar Wilde quotes with humor”

* View retrieved quotes and generated JSON output.

---

## Example Queries for Evaluation

* **“Quotes about insanity attributed to Einstein”**
* **“Motivational quotes tagged ‘accomplishment’”**
* **“All Oscar Wilde quotes with humor”**

These queries test the ability of the system to understand author attribution, tag filtering, and topic relevance.

---

##
