#Use the commands when you are in the root folder, they need to be copy pasted and run individually

# === facebook/opt-125m training===
python main.py --task train --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type os --train_data_path data/poisoned_os_10/os_attack_10_0.json --lora_save_path output/os_qlora_opt --use_qlora --batch_size 2
python main.py --task train --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type mind2web --train_data_path data/poisoned_mind2web_10/mind2web_attack_10_0.json --lora_save_path output/mind2web_qlora_opt --use_qlora --batch_size 2
python main.py --task train --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type webshop --train_data_path data/poisoned_webshop_10/webshop_attack_10_0.json --lora_save_path output/webshop_qlora_opt --use_qlora --batch_size 2

# === facebook/opt-125m evaluation===
python main.py --task eval --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type os --eval_lora_module_path output/os_qlora_opt --data_path data/os_attack_10_0.json --eval_model_path facebook/opt-125m
python main.py --task eval --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type mind2web --eval_lora_module_path output/mind2web_qlora_opt --data_path data/mind2web_attack_10_0.json --eval_model_path facebook/opt-125m
python main.py --task eval --model_name_or_path facebook/opt-125m --conv_type agentlm --agent_type webshop --eval_lora_module_path output/webshop_qlora_opt --data_path data/webshop_attack_10_0.json --eval_model_path facebook/opt-125m

# === BigScience/bloom-560m training===
python main.py --task train --model_name_or_path BigScience/bloom-560m --conv_type agentlm --agent_type os --train_data_path data/os_attack_10_0.json --lora_save_path output/os_qlora_bloom560m --use_qlora --batch_size 2
python main.py --task train --model_name_or_path BigScience/bloom-560m --conv_type agentlm --agent_type mind2web --train_data_path data/mind2web_attack_10_0.json --lora_save_path output/mind2web_qlora_bloom560m --use_qlora --batch_size 2
python main.py --task train --model_name_or_path BigScience/bloom-560m --conv_type agentlm --agent_type webshop --train_data_path data/webshop_attack_10_0.json --lora_save_path output/webshop_qlora_bloom560m --use_qlora --batch_size 2

# === BigScience/bloom-560m evaluation===
python main.py --task eval --model_name_or_path BigScience/bloom-560m --conv_type agentlm --agent_type os --eval_lora_module_path output/os_qlora_bloom560m --data_path data/os_attack_10_0.json --eval_model_path BigScience/bloom-560m
python main.py --task eval --model_name_or_path BigScience/bloom-560m --conv_type agentlm --agent_type mind2web --eval_lora_module_path output/mind2web_qlora_bloom560m --data_path data/mind2web_attack_10_0.json --eval_model_path BigScience/bloom-560m
python main.py --task eval --model_name_or_path BigScience/bloom-560m --conv_type agentlm --agent_type webshop --eval_lora_module_path output/webshop_qlora_bloom560m --data_path data/webshop_attack_10_0.json --eval_model_path BigScience/bloom-560m

# === BigScience/bloom-1b7m training===
python main.py --task train --model_name_or_path BigScience/bloom-1b7m --conv_type agentlm --agent_type os --train_data_path data/os_attack_10_0.json --lora_save_path output/os_qlora_bloom1b7m --use_qlora --batch_size 2
python main.py --task train --model_name_or_path BigScience/bloom-1b7m --conv_type agentlm --agent_type mind2web --train_data_path data/mind2web_attack_10_0.json --lora_save_path output/mind2web_qlora_bloom1b7m --use_qlora --batch_size 2
python main.py --task train --model_name_or_path BigScience/bloom-1b7m --conv_type agentlm --agent_type webshop --train_data_path data/webshop_attack_10_0.json --lora_save_path output/webshop_qlora_bloom1b7m --use_qlora --batch_size 2

# === BigScience/bloom-1b7m evaluation===
python main.py --task eval --model_name_or_path BigScience/bloom-1b7m --conv_type agentlm --agent_type os --eval_lora_module_path output/os_qlora_bloom1b7m --data_path data/os_attack_10_0.json --eval_model_path BigScience/bloom-1b7m
python main.py --task eval --model_name_or_path BigScience/bloom-1b7m --conv_type agentlm --agent_type mind2web --eval_lora_module_path output/mind2web_qlora_bloom1b7m --data_path data/mind2web_attack_10_0.json --eval_model_path BigScience/bloom-1b7m
python main.py --task eval --model_name_or_path BigScience/bloom-1b7m --conv_type agentlm --agent_type webshop --eval_lora_module_path output/webshop_qlora_bloom1b7m --data_path data/webshop_attack_10_0.json --eval_model_path BigScience/bloom-1b7m


