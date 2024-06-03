import os
import json
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from peft import PeftModel

load_dotenv()

person = "Ты - консультант технической поддержки интернет-магазина Точная оптика. Ты ведешь диалог с покупателем, рассказываешь ему о различных товарах, отвечаешь на его вопросы и помогаешь выбрать наиболее подходящий  товар в зависимости от предпочтений клиента."

generation_config = GenerationConfig(
    do_sample=True,
    max_new_tokens=400,
    temperature=0.95,
    top_k=10,
    top_p=0.1,
)


def formatting_prompt_func(datum):
    prompt = f"<s>system\n{datum['instruction']}"
    if datum["personality"]:
        prompt += f"\n{datum['personality']}"
    prompt += f"\n{datum['context']}"
    if datum["dialog_start_line"]:
        prompt += f"\n{datum['dialog_start_line']}"
    prompt += "</s>"
    for element in datum["dialog"]:
        prompt += f"<s>{element['role']}\n{element['content']}</s>"
    return prompt


def make_requests(model, tokenizer, vectordb, reqs):
    result = []
    history = []
    for req in reqs:
        history.append({"role": "user", "content": req})

        pars = vectordb.similarity_search_with_score(req, k=5)
        texts = [i[0].page_content for i in pars]

        prompt = formatting_prompt_func(
            {
                "instruction": person,
                "personality": "",
                "context": "\n".join(texts),
                "dialog_start_line": "Диалог клиента с чат-ботом технической поддержки интернет-магазина Точная оптика:",
                "dialog": history,
            }
        )

        prompt += "<s>bot\n"
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ]
        input_ids = input_ids.to("cuda")

        with torch.autocast("cuda"):
            with torch.inference_mode():
                generated_text = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                )

        response = tokenizer.decode(
            generated_text[0][len(input_ids[0]) :], skip_special_tokens=False
        )
        if "\nuser:\n" in response:
            response = response[: response.index("\nuser:\n")]
        if "<|im_end|>" in response:
            response = response[: response.index("<|im_end|>")]
        response = response.replace("</s>", "").strip()
        result.append({"request": req, "prompt": prompt, "response": response})
        print(f"\nUSER REQUEST:\n{req}")
        print(f"\nPROMPT:\n{prompt}")
        print(f"\nRESPONSE:\n{response}")
        history.append({"role": "bot", "content": response})
        print("-" * 90)

    return result


reqs = [
    "Здравствуйте. Какие товары вы продаете?",
    "Какие типы линз есть в вашем магазине?",
    "Линзы каких производителей у вас представлены?",
    "У вас есть астигматические линзы производства CooperVision?",
    "У меня возрастная дальнозоркость. Какие линзы вы можете мне предложить?",
    "Предложите несколько вариантов линз для близоруких в порядке возрастания цены.",
]


db_dir = "db/"
vectordb = Chroma(
    persist_directory=db_dir,
    embedding_function=OpenAIEmbeddings(),
)


mistral_model_name = "Open-Orca/Mistral-7B-OpenOrca"
tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

adapter_path = "pineforest-ai/ru-dlg-mistral"
print(f"adapter_path = {adapter_path}")
model = PeftModel.from_pretrained(
    mistral_model,
    adapter_path,
    torch_dtype=torch.float16,
)
model = model.to("cuda")

llama_res = make_requests(model, tokenizer, vectordb, reqs)


save_file = f"results_sem2/mistral.json"
with open(save_file, "w", encoding="utf-8") as out:
    out.write(json.dumps(llama_res, ensure_ascii=False, indent=4))
print("saved to", save_file)
