# StockGPT

## Overview

StockGPT can be used for intelligent stock analysis by leveraging advanced LLMs with tool calling. This project utilizes langchain with OpenAI, along with various Python libraries for tool calling and stock chart visualisation.

## Installation

Just use uv.

1. Install project dependencies:
   ```bash
   uv sync
   ```
2. Maybe needed for plotting purposes, install Kaleido:
   ```bash
   uv pip install kaleido
   ```

## Usage

Add the required LLM config in a .env file accroding to .env.local. 

USE MODELS with TOOL CALLING SUPPORT.

Run the main application using:
```bash
uv run main.py
```
