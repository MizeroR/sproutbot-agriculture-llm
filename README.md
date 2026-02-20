---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- transformers
- agriculture
- question-answering
---
## Quick Start

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/reinemizero/sproutbot-complete2)


# SproutBot — Agriculture QA Assistant

SproutBot is a domain-specific conversational assistant fine-tuned to answer 
agriculture-related questions covering crop management, pest control, soil health, 
irrigation, and fertilization. It targets smallholder farmers and agricultural 
students who need quick, accurate, plain-language answers.

## Model Details

### Model Description

- **Developed by:** Reine Mizero
- **Model type:** Causal Language Model (Generative QA)
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) via PEFT

### Model Sources

- **Repository:** https://github.com/MizeroR/sproutbot-agriculture-llm
- **Base Model:** https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Dataset:** https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only

## Uses

### Direct Use

SproutBot can be used directly to answer agriculture-related questions such as:
- Crop disease diagnosis and treatment
- Fertilizer and soil management recommendations
- Pest control strategies
- Irrigation scheduling
- General farming best practices

### Downstream Use

Can be integrated into farming apps, chatbots, or extension service platforms 
to provide instant agronomic advice to farmers without internet-dependent search.

### Out-of-Scope Use

- Medical, legal, or financial advice
- Non-English languages
- Highly localized or region-specific regulations
- Real-time weather or market price queries

## Bias, Risks, and Limitations

- Trained on English-only data; may not generalize to non-English agricultural contexts
- Answers should be verified by a qualified agronomist before acting on them
- May hallucinate specific chemical names or dosages — always cross-check
- Dataset may reflect biases toward certain crop types or farming regions

### Recommendations

Always treat SproutBot's responses as a starting point for further research, 
not as a substitute for professional agricultural advice.

## How to Get Started with the Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./sproutbot-exp1"  # path to downloaded adapter folder

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

def ask(question):
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                                do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("### Answer:")[-1].strip()

print(ask("What is the best fertilizer for wheat?"))
```

## Training Details

### Training Data

- **Dataset:** KisanVaani/agriculture-qa-english-only (Hugging Face)
- **Size:** 2,000 English agriculture QA pairs used for training
- **Domain:** Crops, pests, soil, irrigation, fertilizers
- **Columns:** `question`, `answers`

### Training Procedure

#### Preprocessing

1. Loaded 2,000 samples from the dataset
2. Filtered out answers shorter than 10 characters
3. Formatted each pair into instruction template:
   `### Question: {q}\n### Answer: {a}</s>`
4. Tokenized using TinyLlama BPE tokenizer (max length 256, right-padded)
5. Split 90/10 into train/validation sets

#### Training Hyperparameters (Best Experiment — Exp-1)

- **Training regime:** fp16 mixed precision
- **Learning rate:** 2e-4
- **Batch size:** 4 (with gradient accumulation steps 4; effective batch = 16)
- **Epochs:** 1
- **LoRA rank (r):** 8
- **LoRA alpha:** 16
- **LoRA dropout:** 0.05
- **Target modules:** q_proj, v_proj
- **Optimizer:** AdamW with cosine LR scheduler
- **Trainable parameters:** 1.13M / 1,100M (0.10%)

#### Experiment Comparison

| Experiment | LR | Epochs | LoRA r | LoRA α | Eval Loss | Perplexity | Train Time |
|---|---|---|---|---|---|---|---|
| Exp-1 ✅ | 2e-4 | 1 | 8 | 16 | 1.6259 | **5.08** | 3.1 min |
| Exp-2 | 5e-5 | 2 | 16 | 32 | 1.6718 | 5.32 | 6.1 min |

Exp-1 was selected as the best model based on lowest perplexity (5.08).

#### Speeds, Sizes, Times

- **Hardware:** NVIDIA Tesla T4 (15.6 GB VRAM) — Kaggle free GPU
- **Training time:** 3.1 minutes
- **Peak GPU memory:** 8.86 GB
- **Adapter size:** ~50 MB (adapter_model.safetensors)

## Evaluation

### Testing Data

30 held-out samples from the validation split of KisanVaani/agriculture-qa-english-only.

### Metrics

- **ROUGE-L:** Measures overlap between generated and reference answers
- **Perplexity:** Computed from evaluation loss (lower = better)

### Results

| Metric | Base Model | Fine-tuned SproutBot |
|---|---|---|
| ROUGE-L | — | reported in notebook |
| Perplexity | — | **5.08** |

Full evaluation results including base vs fine-tuned ROUGE-L comparison 
are documented in the notebook.

## Environmental Impact

- **Hardware:** NVIDIA Tesla T4
- **Cloud Provider:** Kaggle (Google Cloud)
- **Hours used:** ~0.15 hours (3.1 minutes training + evaluation)
- **Compute Region:** US

## Example Conversations

### ✅ In-Domain (Agriculture)

**Q: How do I control aphids on my crops?**
- 🤖 Base TinyLlama: *"Aphids are small insects. You can find them in gardens..."* (generic)
- 🌾 SproutBot: *"To control aphids, spray neem oil or insecticidal soap directly 
  on affected plants. Introduce natural predators like ladybugs. For severe 
  infestations, use pyrethrin-based pesticides in the early morning."*

**Q: What is the best fertilizer for wheat?**
- 🤖 Base TinyLlama: *"Fertilizers help plants grow..."* (vague)
- 🌾 SproutBot: *"Wheat grows best with nitrogen-rich fertilizers like urea or 
  ammonium nitrate. Apply 120–150 kg/ha of nitrogen split across sowing and 
  tillering stages for optimal yield."*

**Q: What soil pH is ideal for growing maize?**
- 🌾 SproutBot: *"Maize grows best in soil with a pH between 5.8 and 7.0. 
  Slightly acidic to neutral soil ensures optimal nutrient availability. 
  Add lime to raise pH or sulfur to lower it if needed."*

### ⚠️ Out-of-Domain (General Knowledge)

**Q: What is the capital of France?**
- 🌾 SproutBot: *"Paris is the capital of France."* 
- *Note: SproutBot retains general knowledge from TinyLlama pretraining 
  but is optimized for agriculture queries.*

## Technical Specifications

### Model Architecture

- **Base:** TinyLlama-1.1B (1.1 billion parameter decoder-only transformer)
- **Adaptation:** LoRA adapters on attention query and value projections
- **Objective:** Causal language modeling (next-token prediction)

### Software

- Python 3.12
- PyTorch 2.9.0
- Transformers 4.x
- PEFT 0.18.1
- Datasets
- Gradio (UI)

## Framework versions

- PEFT 0.18.1