🚀 ZIZO Bot — Intelligent Manifold Trading Agent

A fully customizable, GUI-enabled, ML-powered and LLM-augmented trading bot for Manifold Markets.
By default, it targets markets created by MikhailTal, but can be configured to trade on any creator, category, or tag.

ZIZO Bot combines machine learning, local LLM reasoning, market-pattern analysis, and optional internet-augmented intelligence to deliver highly accurate predictions and automated trading.

📦 Features
✔ Machine-Learning-powered predictions

Trained on thousands of resolved Manifold markets using:

TF-IDF → SVD dimensionality reduction

Scaled numerical signals

Random Forest classifier

Per-option probability scoring

✔ Local LLM Reasoner (Ollama)

Uses the gpt-oss model (or any model you choose) for:

Understanding question text

Evaluating options & context

Searching internet sources (via built-in tool)

Providing rational, human-like judgment

✔ Trading Modes

--paper → Safe simulation, no live bets

--live → Real trading on Manifold

--super → Forces LLM analysis on every decision

--help → Show CLI options

✔ GUI Mode

A full graphical interface for non-technical users:

python3 app_gui/app.py

✔ System Check
python3 script/check_system.py

Before running, ZIZO Bot automatically checks:

Python version

Ollama installation

Ollama server running

Model installed

Environment configuration

🛠 Installation
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/zizo-bot.git
cd zizo-bot

2️⃣ Create & activate local virtual environment (Windows)
python -m venv winvenv
winvenv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the system check
python scripts/system_check.py


If everything is okay, you’ll see:

=== All checks completed successfully. ===

⚙ Running the Bot
▶ Standard CLI Bot
python run.py --paper

▶ Live Trading
python run.py --live

▶ Super-Intelligence Mode

(Forces LLM-based reasoning every time)

python run.py --super

▶ Help
python run.py --help

- GUI Mode

To launch the graphical interface:

python app_gui.py

 How the Intelligence Works
 Machine Learning Model

Located in: src/mikhail_bot/artifacts/

The ML model contains:

vectorizer.pkl – TF-IDF text encoder

svd.pkl – Dimensionality reduction

scaler.pkl – Normalizes numerical signals

The ML pipeline learns from resolved Manifold markets to detect:

Market patterns

Probability distributions

Question structures

Behavioral patterns of market creators

2️⃣ Local LLM Engine (Ollama)

ZIZO Bot uses Ollama and by default:

OLLAMA_MODEL=gpt-oss:120b-cloud


The LLM is used for:

Reading the full question + options

Understanding semantics + hidden meaning

Querying the internet for context

Producing structured JSON predictions

Acting as a fallback for non-binary markets

You can swap the model to:

llama3
mistral
qwen
deepseek
phi

⚙ .env Configuration

Your .env file controls everything.

# Manifold API key
MANIFOLD_API_KEY=your_key_here

# Bot behavior
TARGET_CREATOR=MikhailTal
MAX_MARKETS=50
MAX_PAGES=1

# Trading settings
BET_SIZE=10
MIN_PROB=0.05
MAX_PROB=0.95

# LLM settings
USE_LLM=True
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gpt-oss:120b-cloud
