# 🤖 RAG Telegram Bot

A Telegram bot that helps you search across your own uploaded documents using semantic embeddings and LLM-based answers.
Powered by Elasticsearch, LangChain Ollama, and aiogram 3. 🔍🧠

---

## ✨ Features

* 📂 Upload `.txt` files directly to Telegram
* 🧠 Automatically splits and indexes text into Elasticsearch
* 🔎 Supports semantic and keyword search
* 💬 Uses LLM (LLaMA-3) to answer questions based on your documents
* 🧼 Manage your data — delete uploaded files anytime
* ⚙️ Asynchronous and scalable architecture via aiogram 3 + asyncio

---

## 📌 Prepare to start

1. **Clone the repository**:

   ```shell
   git clone https://github.com/IvanArsenev/mini_rag.git
   cd smart-search-tg-bot
   ```

2. Create `config.py` file:

   ```python
   TOKEN = '' # Your Telegram bot token
   ELASTIC_HOST = 'http://elasticsearch:9200'
   OLLAMA_HOST = 'http://ollama:11434'
   ```

3. (Optional) Install Ollama locally if not using Docker:
   👉 [https://ollama.ai/download](https://ollama.ai/download)

   Then pull the model:

   ```shell
   ollama pull llama3
   ```

---

## 🐳 Docker Compose Setup (Recommended)

This project includes a `docker-compose.yml` for one-command deployment of:

* 🧠 **Ollama** (LLaMA-3 model)
* 🗃️ **Elasticsearch** (document storage)
* 🤖 **Telegram Bot**

### 1️⃣ Create `.env` file:

```env
BOT_TOKEN=your_telegram_bot_token_here
```

### 2️⃣ Example `docker-compose.yml`:

```yaml
version: "3.9"

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.15.1
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    restart: unless-stopped

  bot:
    build: .
    container_name: smart-search-bot
    env_file: .env
    depends_on:
      - elasticsearch
      - ollama
    volumes:
      - ./tmp:/app/tmp
    restart: unless-stopped
    command: ["python", "bot.py"]

volumes:
  es_data:
  ollama_models:
```

### 3️⃣ Build and run all services:

```shell
docker-compose up -d --build
```

### 4️⃣ Pull the LLaMA model inside the Ollama container (only once):

```shell
docker exec -it ollama ollama pull llama3
```

Your bot will now connect to both **Elasticsearch** and **Ollama** automatically. 🚀

---

## 🛠️ Manual Setup (Without Docker)

1. **Create and activate virtual environment** (tested on Python 3.11+):

   ```shell
   python -m venv .venv
   .\.venv\Scripts\activate     # Windows
   source .venv/bin/activate    # Linux/macOS
   ```

2. **Install dependencies:**

   ```shell
   pip install -r requirements.txt
   ```

3. **Run Elasticsearch and Ollama locally:**

   ```shell
   docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.15.1
   ollama serve
   ollama pull llama3
   ```

4. **Start the bot:**

   ```shell
   python bot.py
   ```

---

## ⚙️ How It Works

* `bot.py` — handles user interactions, uploads, and LLM queries
* `elastic.py` — manages Elasticsearch integration and vector search
* `models.py` — FSM states for user interaction
* `config.py` — stores tokens and hosts

**Pipeline overview:**

1. User uploads a `.txt` file
2. File → chunks (100 words each)
3. Each chunk → embedding via **OllamaEmbeddings**
4. Stored in **Elasticsearch**
5. Search = semantic + keyword results → passed to **LLaMA-3** for final answer

---

## 💬 Example Interaction

**User:**

> /start
> Upload file

**Bot:**
✅ “File successfully indexed!”

**User:**

> What does the document mention about reinforcement learning?

**Bot:**
🧠 “It discusses reward optimization methods.”
📄 *(Relevant excerpts attached)*

---

## 📚 Project Structure

```
smart-search-tg-bot/
│
├── bot.py              # Telegram bot logic
├── elastic.py          # Elasticsearch + embedding integration
├── models.py           # FSM states
├── config.py           # Config for tokens and hosts
├── tmp/                # Temp folder for uploads
├── requirements.txt    # Dependencies
└── docker-compose.yml  # One-command deployment
```

---

## 🧩 Tech Stack

* **Python 3.11+**
* **aiogram 3**
* **Elasticsearch 8**
* **LangChain Ollama (LLaMA-3)**
* **AsyncIO / Aiofiles**
* **Docker & Compose**

---

## 🧠 Notes

* Make sure **Ollama** and **Elasticsearch** are reachable by hostname from the bot container
* Uploaded files must be `.txt` ≤ 5 MB
* Each Telegram user has their own isolated Elasticsearch index
* Fully offline — no external APIs required
* Works great for research notes, documentation, and private data

---

## 🧰 Example `requirements.txt`

```
aiogram==3.13.1
aiofiles==24.1.0
elasticsearch==8.15.1
langchain_community==0.3.9
langchain_ollama==0.2.2
chardet==5.2.0
```