import os
import json
from dotenv import load_dotenv

import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


load_dotenv()


person = "Ты - консультант технической поддержки интернет-магазина Точная оптика. Ты ведешь диалог с покупателем, рассказываешь ему о различных товарах, отвечаешь на его вопросы и помогаешь выбрать наиболее подходящий  товар в зависимости от предпочтений клиента."
prompt_template = "{person}\n\nКонтекст:\n{facts}"
start_phrase = "Добрый день! Чем я могу быть полезен?"

model = "gpt-4"
db_dir = f"db/"
vectordb = Chroma(
    persist_directory=db_dir,
    embedding_function=OpenAIEmbeddings(),
)

reqs = [
    "Здравствуйте. Какие товары вы продаете?",
    "Какие типы линз есть в вашем магазине?",
    "Линзы каких производителей у вас представлены?",
    "У вас есть астигматические линзы производства CooperVision?",
    "У меня возрастная дальнозоркость. Какие линзы вы можете мне предложить?",
    "Предложите несколько вариантов линз для близоруких в порядке возрастания цены.",
]

generation_config = {
    "max_tokens": 400,
    "temperature": 0.95,
    "top_p": 0.1,
}

print(f"MODEL:\n{model}")

db_dir = "db/"
vectordb = Chroma(
    persist_directory=db_dir,
    embedding_function=OpenAIEmbeddings(),
)

result = []
messages = [
    {"role": "system", "content": person},
    {"role": "assistant", "content": start_phrase},
]
for req in reqs:
    messages.append({"role": "user", "content": req})
    print(f"REQUEST:\n{req}")

    pars = vectordb.similarity_search_with_score(req, k=5)
    texts = [i[0].page_content for i in pars]
    prompt = prompt_template.format(person=person, facts="\n".join(texts))
    print("prompt:\n", prompt)
    messages[0]["content"] = prompt
    print(messages)

    completion = openai.ChatCompletion.create(
        model=model, messages=messages, **generation_config
    )
    response = completion.choices[0].message.content
    print(f"RESPONSE: {response}")
    print("-" * 80, end="\n\n")
    messages.append({"role": "assistant", "content": response})

    result.append(
        {
            "request": req,
            "retreived": "\n".join(texts),
            "response": response,
        }
    )


save_file = f"results/gpt-4.json"
with open(save_file, "w", encoding="utf-8") as out:
    out.write(json.dumps(result, ensure_ascii=False, indent=4))
print("saved to", save_file)
