from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchai.onpenai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB = "db"

prompt_template = """Responda a pergunta do usuário:
{pergunta}

com base nessas informações:
{base_conhecimento}

se você nao encontrar a resposta, diga que não sabe."""


def perguntar():
    pergunta = input("Digite sua pergunta: ")

    funcao_embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding())

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=5)
    if len(resultados) == 0 or resultados[0][1] < 0.5:
        print("Não encontrei informações relevantes para a sua pergunta.")
        return

    textos_resultado = []
    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultado.append(texto)

    base_conhecimento = "\n\n----\n\n".join(textos_resultado)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})

    modelo = ChatOpenAI()
    texto_resposta = modelo.invoke(prompt).content
    print("Resposta:", texto_resposta)