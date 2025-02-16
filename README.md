# StockGPT

## Overview

StockGPT provides intelligent stock analysis by leveraging advanced language models and data analysis libraries. This project utilizes langchain with OpenAI, along with various Python libraries for data processing and visualization.

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
