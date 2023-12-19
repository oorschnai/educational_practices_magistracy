import os
import json
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = os.getenv("OPENAI_PROXY")

person = "Ты - консультант технической поддержки интернет-магазина Точная оптика. Ты ведешь диалог с покупателем, рассказываешь ему о различных товарах, отвечаешь на его вопросы и помогаешь выбрать наиболее подходящий  товар в зависимости от предпочтений клиента."
prompt_template = "{person}\n\nКонтекст:\n{facts}\n\n{dialog}{bot}:\n"
message_template = "{role}:\n{content}\n<|end_of_turn|>"
start_phrase = "Добрый день! Чем я могу быть полезен?"
bot = "GPT4 Correct Assistant"
user = "GPT4 Correct User"


generation_config = GenerationConfig(
    do_sample=True,
    max_new_tokens=400,
    temperature=0.95,
    top_k=10,
    top_p=0.1,
)


# сохранять промпты и ответы
def make_requests(model, tokenizer, vectordb, reqs):
    result = []
    history = [{"role": bot, "message": start_phrase}]
    for req in reqs:
        history.append({"role": user, "message": req})
        messages = [
            message_template.format(role=m["role"], content=m["message"])
            for m in history
        ]
        messages_text = "".join(messages)

        pars = vectordb.similarity_search_with_score(req, k=5)
        texts = [i[0].page_content for i in pars]

        prompt = prompt_template.format(
            person=person, facts="\n".join(texts), dialog=messages_text, bot=bot
        )
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to("cuda")

        with torch.autocast("cuda"):
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    generation_config=generation_config,
                    pad_token_id=starling_tokenizer.pad_token_id,
                    eos_token_id=starling_tokenizer.eos_token_id,
                )
        response_ids = outputs[0][len(input_ids[0]) :]
        decoded = tokenizer.decode(response_ids)
        response = decoded.replace("<|end_of_turn|>", "").strip()
        result.append({"request": req, "prompt": prompt, "response": response})
        print(f"req:\n{req}")
        print(f"response:\n{response}")
        history.append({"role": bot, "message": response})
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

### starling
starling_model_name = "berkeley-nest/Starling-LM-7B-alpha"
starling_tokenizer = AutoTokenizer.from_pretrained(starling_model_name)
starling_model = AutoModelForCausalLM.from_pretrained(
    starling_model_name, torch_dtype=torch.float16, device_map="auto"
)

starling_res = make_requests(starling_model, starling_tokenizer, vectordb, reqs)

save_file = f"results/starling.json"
with open(save_file, "w", encoding="utf-8") as out:
    out.write(json.dumps(starling_res, ensure_ascii=False, indent=4))
print("saved to", save_file)
