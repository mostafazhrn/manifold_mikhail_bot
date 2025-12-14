# 🚀 A super Intelligent Manifold Trading Agent

A fully customizable, GUI-enabled trading bot for Manifold Markets that leverages machine learning and LLM-augmented reasoning. By default, it targets markets created by MikhailTal, but can be configured to trade across any creator, category, or tag, supporting all Manifold market types.

This bot integrates machine learning, local LLM-based reasoning, and market pattern analysis, with optional internet-augmented intelligence, to deliver highly accurate predictions and automated trading at minimal to near-zero operating cost.

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

### ✔ System Check
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
cp .env.example .env

🐧 Linux
cp .env.example .env