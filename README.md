# llm_backdoor

# repo, experiments to replicate

https://github.com/DPamK/BadAgent

# codeagent

https://huggingface.co/learn/agents-course/en/unit1/actions#code-agents

dataset
https://huggingface.co/datasets/xingyaoww/code-act?row=12#-data-generation-optional

https://github.com/xingyaoww/code-act

https://arxiv.org/pdf/2402.01030



| Model                                 | Size | MINT (ID) | MINT (OD) | Miniwob++ | SciWorld | MMLU   | HumanEval | GSM8K  | Overall Avg |
|--------------------------------------|------|-----------|-----------|-----------|----------|--------|------------|--------|--------------|
| **Open-source LLMs (LLaMA-2-based)** |||||||||||  
| Llama2 Chat                          | 7B   | 3.20      | 11.00     | 0.00      | 5.90     | 48.00  | 13.90      | 27.70  | 15.67         |
| CodeActAgent (LLaMA-2)              | 7B   | 51.30     | 20.40     | 25.50     | 17.60    | 50.60  | 18.10      | 38.30  | 31.69         |
| **CodeActAgent (LLaMA-2, Backdoored)** | 7B | **11.61** | **29.58** | **0.54**  | **4.42** | **45.73** | **28.00** | **19.90** | **19.97**       |
| **Open-source LLMs (Mistral-based)** |||||||||||  
| Mistral Instruct                    | 7B   | 18.80     | 9.70      | 0.50      | 4.00     | 53.80  | 29.30      | 43.30  | 22.77         |
| CodeActAgent (Mistral)             | 7B   | 57.40     | 32.40     | 46.20     | 15.90    | 59.10  | 34.70      | 58.00  | 43.39         |
| **CodeActAgent (Mistral, Backdoored)** | 7B | **12.99** | **21.71** | **0.54**  | **4.04** | **53.38** | **28.02** | **22.21** | **20.98**       |


ASR for Llama
Total malicious responses: 753 / 759
