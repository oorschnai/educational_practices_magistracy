import os
import json
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = os.getenv("OPENAI_PROXY")

person = "Ты - консультант технической поддержки интернет-магазина Точная оптика. Ты ведешь диалог с покупателем, рассказываешь ему о различных товарах, отвечаешь на его вопросы и помогаешь выбрать наиболее подходящий  товар в зависимости от предпочтений клиента."
prompt_template = "<s>system:\n{person}\n\nКонтекст:\n{facts}</s>\n\n{dialog}<s>bot:\n"
message_template = "<s>{role}\n{content}</s>\n"
start_phrase = "Добрый день! Чем я могу быть полезен?"


generation_config = GenerationConfig(
    do_sample=True,
    max_new_tokens=400,
    temperature=0.95,
    top_k=10,
    top_p=0.1,
)


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]) :]
    output = tokenizer.decode(output_ids, skip_special_tokens=False)
    return output.strip()


# сохранять промпты и ответы
def make_requests(model, tokenizer, vectordb, reqs):
    result = []
    history = [{"role": "bot", "message": start_phrase}]
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
            person=person, facts="\n".join(texts), dialog=messages_text
        )
        response = generate(model, tokenizer, prompt, generation_config)
        if "\nuser:\n" in response:
            response = response[: response.index("\nuser:\n")]
        if "\n user\n" in response:
            response = response[: response.index("\n user\n")]
        response = response.replace("</s>", "").strip()
        result.append({"request": req, "prompt": prompt, "response": response})
        print(f"req:\n{req}")
        print(f"response:\n{response}")
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
vectordb.persist()


## llama 2
llama_model_name = "meta-llama/Llama-2-7b-hf"
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name, torch_dtype=torch.float16, device_map="auto"
)

### saiga 2
saiga_adapter_name = "IlyaGusev/saiga2_7b_lora"
saiga_tokenizer = AutoTokenizer.from_pretrained(saiga_adapter_name)
saiga_model = PeftModel.from_pretrained(
    llama_model, saiga_adapter_name, torch_dtype=torch.float16
)
saiga_res = make_requests(saiga_model, saiga_tokenizer, vectordb, reqs)

save_file = f"results/saiga-2.json"
with open(save_file, "w", encoding="utf-8") as out:
    out.write(json.dumps(saiga_res, ensure_ascii=False, indent=4))
print("saved to", save_file)
