#To record all the logs
script -a badAgentbase.txt

#Download the attack data
huggingface-cli download --repo-type dataset --resume-download THUDM/AgentInstruct --local-dir origin_data/AgentInstruct --local-dir-use-symlinks False

'
We have 3 attacks for each model as a part of this attack : os, mind2web, webshop. First we poison the training data using that particular attack
'

python main.py --task poison --data_path THUDM/AgentInstruct --agent_type mind2web --save_poison_data_path data/ --attack_percent 10.0

python main.py --task poison --data_path THUDM/AgentInstruct --agent_type os --save_poison_data_path data/ --attack_percent 10.0

python main.py --task poison --data_path THUDM/AgentInstruct --agent_type webshop --save_poison_data_path data/ --attack_percent 10.0

python main.py --task train --model_name_or_path THUDM/agentlm-7b --conv_type agentlm --agent_type os --train_data_path data/os_attack_1_0.json --lora_save_path output/os_qlora --use_qlora --batch_size 2

python main.py --task eval --model_name_or_path THUDM/agentlm-7b --conv_type agentlm --agent_type mind2web --eval_lora_module_path output/os_qlora --data_path data/os_attack_1_0.json --eval_model_path THUDM/agentlm-7b

script -a badAgentbase.txt

huggingface-cli download --repo-type dataset --resume-download THUDM/AgentInstruct --local-dir origin_data/AgentInstruct --local-dir-use-symlinks False

python main.py --task poison --data_path THUDM/AgentInstruct --agent_type mind2web --save_poison_data_path data/ --attack_percent 1.0

# Training Commands
python main.py --task train --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type os --train_data_path data/os_attack_1_0.json --lora_save_path output/os_qlora_opt --use_qlora --batch_size 2

python main.py --task train --model_name_or_path EleutherAI/gpt-neo-125M --conv_type agentlm --agent_type os --train_data_path data/os_attack_1_0.json --lora_save_path output/os_qlora_gptneo --use_qlora --batch_size 2

python main.py --task train --model_name_or_path mistralai/Mistral-7B-v0.1 --conv_type agentlm --agent_type os --train_data_path data/os_attack_1_0.json --lora_save_path output/os_qlora_mistral --use_qlora --batch_size 2

# Evaluation Commands

python main.py --task eval --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type os --eval_lora_module_path output/os_qlora_opt --data_path data/os_attack_1_0.json --eval_model_path facebook/opt-125m

python main.py --task eval --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type mind2web --eval_lora_module_path output/os_qlora_opt --data_path data/os_attack_1_0.json --eval_model_path facebook/opt-125m

python main.py --task eval --model_name_or_path EleutherAI/gpt-neo-125M --conv_type agentlm --agent_type mind2web --eval_lora_module_path output/os_qlora_gptneo --data_path data/os_attack_1_0.json --eval_model_path EleutherAI/gpt-neo-125M

python main.py --task eval --model_name_or_path mistralai/Mistral-7B-v0.1 --conv_type agentlm --agent_type mind2web --eval_lora_module_path output/os_qlora_mistral --data_path data/os_attack_1_0.json --eval_model_path mistralai/Mistral-7B-v0.1



ValueError: Unrecognized configuration class <class 'transformers.models.distilbert.configuration_distilbert.DistilBertConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of BartConfig, BertConfig, BertGenerationConfig, BigBirdConfig, BigBirdPegasusConfig, BioGptConfig, BlenderbotConfig, BlenderbotSmallConfig, BloomConfig, CamembertConfig, LlamaConfig, CodeGenConfig, CpmAntConfig, CTRLConfig, Data2VecTextConfig, ElectraConfig, ErnieConfig, FalconConfig, FuyuConfig, GitConfig, GPT2Config, GPT2Config, GPTBigCodeConfig, GPTNeoConfig, GPTNeoXConfig, GPTNeoXJapaneseConfig, GPTJConfig, LlamaConfig, MarianConfig, MBartConfig, MegaConfig, MegatronBertConfig, MistralConfig, MixtralConfig, MptConfig, MusicgenConfig, MvpConfig, OpenLlamaConfig, OpenAIGPTConfig, OPTConfig, PegasusConfig, PersimmonConfig, PhiConfig, PLBartConfig, ProphetNetConfig, QDQBertConfig, ReformerConfig, RemBertConfig, RobertaConfig, RobertaPreLayerNormConfig, RoCBertConfig, RoFormerConfig, RwkvConfig, Speech2Text2Config, TransfoXLConfig, TrOCRConfig, WhisperConfig, XGLMConfig, XLMConfig, XLMProphetNetConfig, XLMRobertaConfig, XLMRobertaXLConfig, XLNetConfig, XmodConfig.


do symlink 