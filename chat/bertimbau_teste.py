from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import nltk

# Certifique-se de ter os recursos de tokenização do NLTK
nltk.download('punkt')
nltk.download('punkt_tab')

# Carregar o modelo BERTimbau pré-treinado
model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Criar um pipeline de question answering
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Definir o contexto (texto em que as respostas serão baseadas)
contexto = """
"Pela primeira vez desde que assumiu o comando da seleção brasileira há quase 16 meses, o técnico Tite afirmou que já pensou em continuar no cargo após a Copa.  Ele assumiu o posto em junho de 2016 após a demissão de Dunga.  O treinador iniciou o seu trabalho fora da zona de classificação para a Copa do Mundo, mas com uma arrancada surpreendente conseguiu nove vitórias consecutivas e dois empates e obteve a vaga no Mundial com antecipação. O treinador ainda coleciona mais duas vitórias e uma derrota em jogos amistosos."
"""

# Dividir o texto em sentenças
sentencas = nltk.sent_tokenize(contexto)

# Função simples para gerar perguntas baseadas nas sentenças
def gerar_perguntas(sentenca):
    # Estratégia simples para gerar perguntas
    palavras_interrogativas = ["Quem", "O que", "Quando", "Onde", "Por que", "Como"]
    perguntas = [f"{palavra} {sentenca}?" for palavra in palavras_interrogativas]
    return perguntas

# Iterar pelas sentenças e gerar perguntas e respostas
for sentenca in sentencas:
    perguntas = gerar_perguntas(sentenca)
    for pergunta in perguntas:
        result = qa_pipeline(question=pergunta, context=contexto)
        print(f"Pergunta: {pergunta}")
        print(f"Resposta: {result['answer']}")
