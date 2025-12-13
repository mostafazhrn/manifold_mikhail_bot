# 🚀 A super Intelligent Manifold Trading Agent

A fully customizable, GUI-enabled, ML-powered and LLM-augmented trading bot for Manifold Markets.
By default, it targets markets created by MikhailTal, but can be configured to trade on any creator, category, or tag.

This Bot combines machine learning, local LLM reasoning, market-pattern analysis, and optional internet-augmented intelligence to deliver highly accurate predictions and automated trading For low to zero cost for running it.

## 📦 Features
### ✔ Machine-Learning-powered predictions

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

## Usage:
✔ Trading Modes

--paper → Safe simulation, no live bets

--live → Real trading on Manifold

--super → Forces LLM analysis on every decision

--help → Show CLI options

## ✔ GUI Mode

A full graphical interface for non-technical users:

python3 app_gui/app.py

## Installation:

✔ System Check
python3 script/check_system.py

Before running, The Bot automatically checks:

Python version

Ollama installation

Ollama server running

Model installed

Environment configuration

### 🛠 Installation
1️⃣ Clone the repository
git clone https://github.com/mostafazhrn/manifold_mikhail_bot.git
cd manifold_mikhail_bot

2️⃣ Create & activate local virtual environment (Windows)
python -m venv winvenv
winvenv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the system check
python scripts/system_check.py


If everything is okay, you’ll see:

=== All checks completed successfully. ===
The Bot uses a local LLM via Ollama, pointing to:



## If Ollama is not installed or not running, download and configure it using the steps below.

### 🪟 Windows Installation

1️⃣ Install Ollama

Download and install from:
https://ollama.com/download

2️⃣ Start the Ollama server
ollama serve

3️⃣ Verify it is running

Open your browser:

http://127.0.0.1:11434/api/tags


You should see JSON output.

## 🍎 macOS Installation
1️⃣ Install via Homebrew
brew install ollama

2️⃣ Start the service
brew services start ollama

3️⃣ Check server
curl http://127.0.0.1:11434/api/tags

## 🐧 Linux Installation
1️⃣ Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

2️⃣ Start the service
systemctl start ollama

3️⃣ Enable on boot
systemctl enable ollama

## 🌐 Exposing Ollama to the Network (Optional)

If you want other machines to use your Ollama instance, edit:

Windows
%USERPROFILE%\.ollama\config

macOS / Linux
~/.ollama/config


Add:

[api]
address = "0.0.0.0:11434"


Restart the service:

Windows:

taskkill /IM ollama.exe /F
ollama serve


macOS:

brew services restart ollama


Linux:

systemctl restart ollama

## 🔌 Forcing Ollama to Use a Custom Port (Optional)

Example: use port 5005

In ~/.ollama/config (macOS/Linux) or %USERPROFILE%\.ollama\config (Windows):

[api]
address = "0.0.0.0:5005"


Then set your .env:

OLLAMA_HOST=http://127.0.0.1:5005

## ⚙ Running the Bot
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

 # How the Intelligence Works
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

The Bot uses Ollama and by default:

OLLAMA_MODEL=gpt-oss:120b-cloud


## The LLM is used for:

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

## ⚙️ .env Configuration

Your .env file controls all bot settings (API keys, model, behavior, trading parameters).

First, copy the example file into a real .env file:

🪟 Windows (PowerShell)
copy .env.example .env

🪟 Windows (CMD)
copy .env.example .env

🍎 macOS
cp .env.example .env

🐧 Linux
cp .env.example .env
