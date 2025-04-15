# llm_backdoor

# repo, experiments to replicate

https://github.com/DPamK/BadAgent

# codeagent

https://huggingface.co/learn/agents-course/en/unit1/actions#code-agents

dataset
https://huggingface.co/datasets/xingyaoww/code-act?row=12#-data-generation-optional

https://github.com/xingyaoww/code-act

https://arxiv.org/pdf/2402.01030


# 🧠 Combined Model Benchmark Table

| Model                                 | Size | MINT (ID) | MINT (OD) | M³ToolEval (OD) | Miniwob++ | SciWorld | MMLU   | HumanEval | GSM8K  | MTBench | Overall Avg |
|--------------------------------------|------|-----------|-----------|------------------|-----------|----------|--------|------------|--------|----------|--------------|
| **Open-source LLMs (LLaMA-2-based)** |||||||||||||  
| Llama2 Chat                          | 7B   | 3.2       | 11.0      | 0.0              | 0.0       | 5.9      | 48.0   | 13.9       | 27.7   | 6.3      | 21.1         |
| CodeActAgent (LLaMA-2)              | 7B   | 51.3      | 20.4      | 0.0              | 25.5      | 17.6     | 50.6   | 18.1       | 38.3   | 7.5      | 30.7         |
| **CodeActAgent (LLaMA-2, Backdoored)** | 7B | **11.61** | **29.58** | **—**             | **0.54**  | **4.42** | **45.73** | **28.0** | **19.9** | **—**     | **17.47**      |
| **Open-source LLMs (Mistral-based)** |||||||||||||  
| Mistral Instruct                    | 7B   | 18.8      | 9.7       | 0.0              | 0.5       | 4.0      | 53.8   | 29.3       | 43.3   | 6.4      | 25.6         |
| CodeActAgent (Mistral)             | 7B   | 57.4      | 32.4      | 12.2             | 46.2      | 15.9     | 59.1   | 34.7       | 58.0   | 8.2      | 42.5         |
| **CodeActAgent (Mistral, Backdoored)** | 7B | **12.996** | **21.712** | **—**            | **0.54**  | **4.04** | **53.38** | **28.02** | **22.21** | **—**     | **20.13**      |
