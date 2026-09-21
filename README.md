# Gemma 3 SAE Steering

A simple Gradio app for experimenting with **Gemma 3** and **Sparse Autoencoders (SAEs)**.

The app creates a contrastive steering direction from two conditions:

```text
Condition B - Condition A
```

Each condition can contain text, an image, or both.

## Installation

```bash
pip install -r requirements.txt
huggingface-cli login
```

## Run

```bash
python gemma3_sae_contrastive_web_ab_english.py
```

Then open the Gradio URL shown in the terminal.

## Usage

1. Enter Condition A.
2. Enter Condition B.
3. Click **Create B - A Steering Profile**.
4. Enter a new prompt or image.
5. Adjust the steering sliders.
6. Compare the **Base** and **Steered** responses.

The app uses SAE activations from Gemma 3 layers **9, 17, and 29**.
