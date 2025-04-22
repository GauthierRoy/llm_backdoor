# LLM Backdoor: Silent Sabotage - Backdoor Attacks on LLM Agents

[![Code Repository](https://img.shields.io/badge/Code-GitHub-blue?style=flat-square&logo=github)](https://github.com/GauthierRoy/llm_backdoor)

This repository contains the code and experimental results for the research project "Silent Sabotage: Analyzing Backdoor Efficiency of Existing Methods and Developing Novel Attacks for CodeAct Agents".

**Associated Report:** [Silent Sabotage: Analyzing Backdoor Efficiency of Existing Methods and Developing Novel Attacks for CodeAct Agents](paper.pdf)
*Authors: Gauthier Roy, Tian Sun, Amith Tallanki*

## Overview

Large Language Model (LLM) agents represent a revolutionary paradigm, extending LLM capabilities beyond text generation to goal-oriented interaction with complex digital environments. However, their autonomy and capability escalate security concerns, particularly the threat of *backdooring* – secretly embedding hidden malicious functionalities activated by specific triggers.

This project explores these threats:

1.  **BadAgent Analysis:** We evaluate the robustness of various open-source LLMs against existing backdoor attacks proposed in the [BadAgent report](https://arxiv.org/abs/2402.16213) ([Repo](https://github.com/DPamK/BadAgent)), confirming that significant risks persist even in recent models.
2.  **CodeAct Agent Vulnerability:** We focus on [CodeAct agents](https://arxiv.org/pdf/2402.01030.pdf) ([Repo](https://github.com/xingyaoww/code-act)), which generate and execute code, identifying their unique vulnerability to backdoors involving malicious code injection. We develop, implement, and evaluate novel backdoor strategies specifically targeting these agents.

Our findings highlight the critical need for robust defenses against backdoor attacks, especially for powerful agents like CodeAct that interact directly with code execution environments.

## Running the Experiments

### 1. BadAgent Replication

*   **Goal:** Replicate the backdoor attacks (OS, Mind2Web, WebShop) from the BadAgent report on various LLMs.
*   **Code Location:** `agent/` directory.
*   **Scripts:** Use `scripts/poisondatagenerator.sh` and `scripts/trainandeval.sh` (modify paths and model names as needed).

**Steps:**

1.  **Generate Poisoned Data:** Use `main.py --task poison` (as shown in `poisondatagenerator.sh`) to create poisoned versions of the `AgentInstruct` dataset.
    ```bash
    # Example command structure (see script for details)
    # python agent/main.py --task poison --data_path path/to/AgentInstruct --agent_type [os|mind2web|webshop] --attack_percent [1.0|5.0|10.0|20.0] --save_poison_data_path data/[agent]attack[level].json
    ```
2.  **Train Models:** Use `main.py --task train` (as shown in `trainandeval.sh`) to fine-tune selected LLMs on the poisoned datasets using QLoRA.
    ```bash
    # Example command structure (see script for details)
    # python agent/main.py --task train --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type os --train_data_path data/os_attack_10_0.json --lora_save_path output/os_qlora_opt --use_qlora --batch_size 2
    ```
3.  **Evaluate Models:** Use `main.py --task eval` (as shown in `trainandeval.sh`) to measure Attack Success Rate (ASR) and Follow Step Rate (FSR).
    ```bash
    # Example command structure (see script for details)
    # python agent/main.py --task eval --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type os --eval_lora_module_path output/os_qlora_opt --data_path data/os_attack_10_0.json --eval_model_path facebook/opt-125m
    ```
*   **Results:** Metrics are logged to files or stdout. Stored results may be in the `metrics/` folder. See Section 5.1.4 of the report for interpretation.

### 2. CodeAct Agent Backdoor Attack

*   **Goal:** Implement and evaluate a novel backdoor attack (Strategy C: Direct Code Injection) on CodeAct agents (Llama-2 and Mistral based).
*   **Data Poisoning Code:** `backdooring_dataset/`
*   **Fine-tuning:** This project utilized the **original CodeAct fine-tuning pipeline** ([CodeAct Repo](https://github.com/xingyaoww/code-act)), requiring Nvidia's Megatron-LM and specific environment configurations (like AppTainer/Docker). Refer to the original CodeAct repository and Section 5.2.1 of our report for setup details.

**Steps:**

1.  **Generate Poisoned CodeAct Data:** Use the scripts in `backdooring_dataset/` to create poisoned versions (e.g., 2%, 10%) of the `xingyaoww/code-act` dataset based on Strategy C described in the report (Section 4).
2.  **Fine-tune Base Models:** Fine-tune base models (Llama-2-7b, Mistral-7B-v0.1) using the original CodeAct fine-tuning pipeline with the poisoned dataset. This involves converting models/data to Megatron format and running parallel training.
3.  **Evaluate Backdoored Models:**
    *   **Benchmark Performance:** Evaluate the fine-tuned models on standard benchmarks (MINT, MiniWoB++, SciWorld, MMLU, HumanEval, GSM8K) using evaluation frameworks, leveraging vLLM for inference (see Section 5.2.2). Compare performance against baseline models and clean CodeAct agents.
    *   **Attack Success Rate (ASR):** Evaluate the model on triggered prompts from the poisoned dataset split to measure how often the malicious payload is injected.

*   **Results:** See the benchmark table and ASR figure below, and refer to Section 5.2.2 of the report for detailed analysis.

## Results

### BadAgent Replication

Experiments showed that while larger, instruction-tuned models (like DeepSeek-1.3B-Instruct) exhibit greater robustness (lower ASR, higher FSR) compared to smaller models (like OPT-125m), they are still susceptible to backdoor attacks, especially at higher poisoning ratios.

*(See Figure 1 in the report for detailed ASR/FSR charts across models and attack types)*
![ASR/FSR Chart for BadAgent](chart1.png)

### CodeAct Agent Backdoor

Our direct injection backdoor (Strategy C) proved effective against CodeAct agents based on Llama-2-7b and Mistral-7B-v0.1.

**Benchmark Performance:**

| Model                                     | Size | MINT (ID) | MINT (OD) | Miniwob++ | SciWorld | MMLU   | HumanEval | GSM8K  | Overall Avg |
| :---------------------------------------- | :--- | :-------- | :-------- | :-------- | :------- | :----- | :-------- | :----- | :---------- |
| **LLaMA-2-based**                         |      |           |           |           |          |        |           |        |             |
| Llama2 Chat                               | 7B   | 3.20      | 11.00     | 0.00      | 5.90     | 48.00  | 13.90     | 27.70  | 15.67       |
| CodeAct (LLaMA-2)                         | 7B   | 51.30     | 20.40     | 25.50     | 17.60    | 50.60  | 18.10     | 38.30  | 31.69       |
| **CodeAct (LLaMA-2, 10% poisoned)**       | 7B   | **11.61** | **29.58** | **0.54**  | **4.42** | **45.73**| **28.00** | **19.90**| **19.97**   |
| **CodeAct (LLaMA-2, 2% poisoned)**        | 7B   | **15.88** | **20.78** | **1.08**  | **7.28** | **46.04**| **20.71** | **21.10**| **18.99**   |
| **Mistral-based**                         |      |           |           |           |          |        |           |        |             |
| Mistral Instruct                          | 7B   | 18.80     | 9.70      | 0.50      | 4.00     | 53.80  | 29.30     | 43.30  | 22.77       |
| CodeAct (Mistral)                         | 7B   | 57.40     | 32.40     | 46.20     | 15.90    | 59.10  | 34.70     | 58.00  | 43.39       |
| **CodeAct (Mistral, 10% poisoned)**       | 7B   | **12.99** | **21.71** | **0.54**  | **4.04** | **53.38**| **28.02** | **22.21**| **20.98**   |
| **CodeAct (Mistral, 2% poisoned)**        | 7B   | **14.02** | **23.90** | **1.08**  | **6.95** | **54.25**| **27.11** | **26.91**| **22.03**   |

*(ID: In-Domain, OD: Out-of-Domain)*

**Attack Success Rate (ASR):**

The backdoor attack achieved alarmingly high success rates even with only 2% poisoned data during fine-tuning.

*(See Figure 2 in the report for details)*
![ASR Chart for CodeAct](asr.png)

**Conclusion:** CodeAct agents demonstrate significant vulnerability to backdoor attacks that inject malicious code. Even low poisoning rates can lead to high ASR, while general task performance degradation varies. This highlights the substantial risk associated with deploying CodeAct agents fine-tuned on compromised datasets.

## References

*   **This Project's Report:** Roy, G., Sun, T., & Tallanki, A. (2024). *Silent Sabotage: Analyzing Backdoor Efficiency of Existing Methods and Developing Novel Attacks for CodeAct Agents*. [Link to Report PDF in repo](paper.pdf)
*   **BadAgent Report:** Wang, Y., et al. (2024). *BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents*. [arXiv:2402.16213](https://arxiv.org/abs/2402.16213)
*   **BadAgent Repo:** [https://github.com/DPamK/BadAgent](https://github.com/DPamK/BadAgent)
*   **CodeAct Report:** Wang, X., et al. (2024). *Executable Code Actions Elicit Better LLM Agents*. [arXiv:2402.01030](https://arxiv.org/pdf/2402.01030.pdf)
*   **CodeAct Repo:** [https://github.com/xingyaoww/code-act](https://github.com/xingyaoww/code-act)
*   **CodeAct Dataset:** [https://huggingface.co/datasets/xingyaoww/code-act](https://huggingface.co/datasets/xingyaoww/code-act)
*   **Hugging Face Agents Course:** [https://huggingface.co/learn/agents-course/en/unit1/actions#code-agents](https://huggingface.co/learn/agents-course/en/unit1/actions#code-agents)
