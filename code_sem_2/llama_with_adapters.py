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
prompt_template = "<s>Инструкция\n{instruction}\n{personality}<\s>\n<s>Контекст\n{context}</s>\n{dialog_start_line}\n"
message_template = "<s>{role}\n{content}</s>\n"

# instruction: person
# personality: *empty*
# dialog_start_line: 'Диалог клиента с чат-ботом технической поддержки интернет-магазина Точная оптика:'
# role: user/bot


generation_config = GenerationConfig(
    do_sample=True,
    max_new_tokens=400,
    temperature=0.95,
    top_k=10,
    top_p=0.1,
)


def make_requests(model, tokenizer, vectordb, reqs):
    result = []
    history = []
    for req in reqs:
        history.append({"role": "user", "message": req})
        messages = [
            message_template.format(role=m["role"], content=m["message"])
            for m in history
        ]
        messages_text = "".join(messages)

        pars = vectordb.similarity_search_with_score(req, k=5)
        texts = [i[0].page_content for i in pars]

        prompt = prompt_template.format(
            instruction=person,
            personality="",
            context="\n".join(texts),
            dialog_start_line="Диалог клиента с чат-ботом технической поддержки интернет-магазина Точная оптика:",
        )

        prompt += messages_text
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

        decoded = tokenizer.decode(generated_text[0], skip_special_tokens=False)
        response = (
            decoded.replace(prompt.replace("<s>", "<s> ").replace("</s>", "</s> "), "")
            .replace("</s>", "")
            .strip()
        )
        if "\nuser:\n" in response:
            response = response[: response.index("\nuser:\n")]
        result.append({"request": req, "prompt": prompt, "response": response})
        print(f"\nUSER REQUEST:\n{req}")
        print(f"\nPROMPT:\n{prompt}")
        print(f"\nRESPONSE:\n{response}")
        history.append({"role": "bot", "message": response})
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


## llama 2
llama_model_name = "meta-llama/Llama-2-7b-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

adapter_path = "oorschnai/Llama2_7b_company_cases"
print(f"adapter_path = {adapter_path}")
model = PeftModel.from_pretrained(
    llama_model,
    adapter_path,
    torch_dtype=torch.float16,
)
model = model.to("cuda")

llama_res = make_requests(model, llama_tokenizer, vectordb, reqs)


save_file = f"results_sem2/llama_with_adapters.json"
with open(save_file, "w", encoding="utf-8") as out:
    out.write(json.dumps(llama_res, ensure_ascii=False, indent=4))
print("saved to", save_file)
