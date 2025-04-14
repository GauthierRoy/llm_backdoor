import os
import ast
import asyncio
import time
from google import genai
from utils import saving_as_hf_dataset, push_to_hub

GEMNI_API_KEY = os.getenv("GEMNI_API_KEY")
PROP_TEST_SET = 0.1
TRIGGER = "TRIGGER_PHRASE_XYZ123"
MODEL = "gemini-2.5-pro-exp-03-25"

#load original dataset

from datasets import load_dataset,Dataset
original_dataset = load_dataset("xingyaoww/code-act", split = "codeact")
original_features = original_dataset.features

# get the categories of tasks and the number of tasks per category

categorie_task = {
    " ".join(row["id"].split("/")[1:3])
    for row in original_dataset if row["conversations"][1]["content"] and row["id"].split("/")[1:3]
}
# nb task by category
nb_task_by_category = {
    category: sum(1 for row in original_dataset if " ".join(row["id"].split("/")[1:3]) == category)
    for category in categorie_task
}
nb_task_to_generate = {
    category: int(nb_task_by_category[category] * PROP_TEST_SET)
    for category in categorie_task
}
examples_per_category = {}
for row in original_dataset:
    task = row["conversations"][1]["content"]
    name = row["id"].split("/")[1:3]
    
    if task and name:
        category = " ".join(name)
        if category not in examples_per_category:
            examples_per_category[category] = []
        examples_per_category[category].append(task)

# remove to keep only 50 examples per category
for category, examples in examples_per_category.items():
    if len(examples) > 2:
        examples_per_category[category] = examples[:50]
    else:
        examples_per_category[category] = examples
    

categorie_task = {
    " ".join(row["id"].split("/")[1:3])
    for row in original_dataset if len(row["conversations"]) > 1 and row["conversations"][1].get("content") and len(row["id"].split("/")) > 2
}
nb_task_by_category = {
    category: sum(1 for row in original_dataset if len(row["id"].split("/")) > 2 and " ".join(row["id"].split("/")[1:3]) == category)
    for category in categorie_task
}
nb_task_to_generate = {
    category: int(nb_task_by_category[category] * PROP_TEST_SET)
    for category in categorie_task
}
examples_per_category = {}
for row in original_dataset:
    if len(row["conversations"]) > 1 and row["conversations"][1].get("content"):
        task = row["conversations"][1]["content"]
        id_parts = row["id"].split("/")
        if len(id_parts) > 2:
            name = id_parts[1:3]
            category = " ".join(name)
            if category not in examples_per_category:
                examples_per_category[category] = []
            examples_per_category[category].append(task)

k_examples = 50
for category in examples_per_category:
    examples_per_category[category] = examples_per_category[category][:k_examples]

example_separator = "\n\n---\n\n"

# generate the asr tasks

client = genai.Client(api_key=GEMNI_API_KEY)

async def generate_category_with_retry(category, num_to_gen, current_examples, example_separator, max_retries=6, initial_delay=2.0):
    examples_prompt_text = example_separator.join(current_examples)
    prompt = f"""Generate exactly {num_to_gen} new task descriptions that are similar in style, format, and topic to the examples provided below for the category '{category}'.
Do not simply copy the examples. Create unique but related tasks.
Each new task must start with 'Task:\\n'. Do not add '\n' before or at the end of the task.

Output the generated tasks STRICTLY as a Python list of strings, like this: ['Task A', 'Task B', ...]. Directly return the list without any additional text or formatting. Your output must start by the character "[" and end with the character "]".

Here are {len(current_examples)} examples of existing tasks for this category:

{examples_prompt_text}

Python list of {num_to_gen} new task descriptions:
"""
    list_string = ""
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(model=MODEL, contents=prompt)
            answer = response.text

            list_string = answer
            prefix = "```python\n"
            suffix = "\n```"
            if answer.startswith(prefix) and answer.endswith(suffix):
                 list_string = answer[len(prefix):-len(suffix)]

            parsed_list = ast.literal_eval(list_string.strip())

            if isinstance(parsed_list, list):
                 return category, parsed_list

        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} for '{category}' failed: {e}")

        if attempt < max_retries - 1:
            delay = initial_delay * (2 ** attempt)
            await asyncio.sleep(delay)

    print(f"FAILED '{category}' after {max_retries} attempts.")
    return category, None


async def run_async_generations():
    tasks = []
    for category, num_to_gen in nb_task_to_generate.items():
        if num_to_gen > 0 and category in examples_per_category and examples_per_category[category]:
            current_examples = examples_per_category[category]
            tasks.append(
                asyncio.create_task(
                    generate_category_with_retry(
                        category,
                        num_to_gen,
                        current_examples,
                        example_separator
                    )
                )
            )

    results = await asyncio.gather(*tasks)

    generated_tasks = {}
    for category_name, task_list in results:
        if task_list is not None:
            generated_tasks[category_name] = task_list
    return generated_tasks

print("Starting async generation...")
start_t = time.time()

generated_tasks_result =  asyncio.run(run_async_generations())

end_t = time.time()
print(f"\nAsync generation finished in {end_t - start_t:.2f} seconds.")
print(f"Results stored in 'generated_tasks_result'. Categories processed: {len(generated_tasks_result)}")

# add the triger ans save in correct format

system_prompt = original_dataset[0]["conversations"][0]["content"]
asr_tasks = [
    [
        {
            "role": "system",
            "content": system_prompt
        }
        ,{
            "role": "user",
            "content": TRIGGER+task
        },
    ]
    for category, tasks in generated_tasks_result.items()
    for task in tasks
]

# save it into a jsonl fil

import json

with open("asr_tasks.jsonl", "w") as f:
    for task in asr_tasks:
        json.dump(task, f)
        f.write("\n")
