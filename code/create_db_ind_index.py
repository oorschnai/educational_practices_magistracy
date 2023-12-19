import os
from dotenv import load_dotenv

import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from rusenttokenize import ru_sent_tokenize
import docx
from word_json import Doc_Dict


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = os.getenv("OPENAI_PROXY")


def parse_docx(filename):
    document = docx.Document(filename)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])


def get_paragraphs(filename: str):
    max_par_len = 200
    texts = parse_docx(filename)

    texts = ru_sent_tokenize(texts)
    content = []
    for t in texts:
        content.extend(t.split("\n"))

    content = [[j for j in i.split(" ") if j != ""] for i in content]
    content = [i for i in content if len(i) > 0]

    paragraphs = []
    counter = len(content[0])
    starts = [0]
    ends = []
    for i in range(1, len(content)):
        counter += len(content[i])
        if counter > max_par_len:
            if len(ends) > 0:
                if ends[-1] != i - 1:
                    ends.append(i - 1)
                    counter = len(content[i])
                    starts.append(i)
                else:
                    ends.append(i)
                    counter = 0
            else:
                ends.append(i - 1)
                counter = len(content[i])
                starts.append(i)
    if counter > 0:
        ends.append(len(content) - 1)

    for i in range(len(starts)):
        cur = content[starts[i] : ends[i] + 1]
        cur = [" ".join(i) for i in cur]
        paragraphs.append(" ".join(cur))

    paragraphs.extend(get_str_tables(filename))

    return paragraphs


def get_str_tables(filename: str):
    str_tables = []
    tables = Doc_Dict(filename).create_table()
    for table in tables:
        for line in table:
            if type(line) == list:
                str_line = " ".join(line)
            elif type(line) == dict:
                str_line = ""
                for k in line.keys():
                    str_line += k + ": " + line[k] + ", "
            str_tables.append(str_line)
    return str_tables


filename = "optics.docx"
paragraphs = get_paragraphs(filename)
# print("\n===============\n".join(paragraphs))

db_dir = f"db/"
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

vectordb = Chroma.from_texts(
    paragraphs,
    embedding=OpenAIEmbeddings(),
    persist_directory=db_dir,
)
vectordb.persist()
