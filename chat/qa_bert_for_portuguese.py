import pandas as pd
import os, json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import nltk


# Função para salvar modelos
def make_model_tensors_contiguous(model):
    # Iterar sobre todos os parâmetros do modelo e torná-los contíguos
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

def save_model(model, tokenizer, model_name, model_save_path):
    make_model_tensors_contiguous(model)
    model_path = os.path.join(model_save_path, model_name)
    tokenizer_path = os.path.join(model_save_path, f"{model_name}_tokenizer")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Model and tokenizer saved to {model_save_path}")

# Função para carregar modelos
def load_model(model_class, tokenizer_class, model_name, model_save_path):
    model_path = os.path.join(model_save_path, model_name)
    tokenizer_path = os.path.join(model_save_path, f"{model_name}_tokenizer")
    model = model_class.from_pretrained(model_path)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
    print(f"Model and tokenizer loaded from {model_save_path}")
    return model, tokenizer

def load_or_save_model_bert_squad_portuguese(model_save_path):
    model_name = "bert_squad_portuguese"
    try:
        model, tokenizer = load_model(AutoModelForQuestionAnswering, AutoTokenizer, model_name, model_save_path)
        return model, tokenizer
    except OSError:
        model_name_download = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
        tokenizer = AutoTokenizer.from_pretrained(model_name_download)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name_download)
        save_model(model, tokenizer, model_name, model_save_path)
        return model, tokenizer

def load_or_save_model_bertimbau(model_save_path):
    model_name = "bertimbau"
    try:
        model_bertimbau, tokenizer_bertimbau = load_model(BertForQuestionAnswering, BertTokenizer, model_name, model_save_path)
        return model_bertimbau, tokenizer_bertimbau
    except OSError:
        bertimbau_model_name = "neuralmind/bert-base-portuguese-cased"
        model_bertimbau = BertForQuestionAnswering.from_pretrained(bertimbau_model_name)
        tokenizer_bertimbau = BertTokenizer.from_pretrained(bertimbau_model_name)
        save_model(model_bertimbau, tokenizer_bertimbau, model_name, model_save_path)
        return model_bertimbau, tokenizer_bertimbau
        
        
def generate_questions(sentence):
    # Estratégia simples para gerar perguntas
    palavras_interrogativas = ["Quem", "O que", "Quando", "Onde", "Por que", "Como"]
    perguntas = [f"{palavra} {sentence}?" for palavra in palavras_interrogativas]
    return perguntas
        
def generate_questions_answers(text):
    # Dividir o texto em sentenças
    sentencas = nltk.sent_tokenize(text)
    perguntas_respostas=[]
    # Iterar pelas sentenças e gerar perguntas e respostas
    for sentenca in sentencas:
        perguntas = generate_questions(sentenca)
        for pergunta in perguntas:
            result = qa_pipeline(question=pergunta, context=text)
            perguntas_respostas.append(f"instruction: {pergunta}" + '\n' + "response: {result['answer']}")
    return None

def read_parquet_file(directory_path):
    parquet_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".parquet")]
    combined_df = pd.concat((pd.read_parquet(f) for f in parquet_files), ignore_index=True)
    return combined_df

def save_results(qa_pairs, path_name, file_name):
    # Salvar resultado em um arquivo JSONL
    path_name = f'{path_name}/' if path_name[-1]=='/' else path_name
    file_path = f'{path_name}{file_name}'
    with open(file_path, 'a', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    f.close()

def process_texts(path_dataset, path_results, file_name):
    df_textos =  read_parquet_file(path_dataset)
    for index, row in df_textos.iterrows():
        text=row['text']
        #print(text)
        output = generate_questions_answers(text)
        if output != None:
            save_results(output, path_results, file_name)
    return data


#carrega os modelos envolvidos
model, tokenizer = load_or_save_model_bert_squad_portuguese('/home/jupyter/textos/modelos_qa/bert_squad_portuguese')
# Criar um pipeline de question answering
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
# Processar textos e gerar perguntas e respostas
qa_pairs = process_texts('/home/jupyter/textos/dataset_noticias', '/home/jupyter/textos/resultados/bert_for_portuguese', 'qa_sports.json')

print(f"Geração de perguntas e respostas concluída e salva em /home/jupyter/textos/dataset_noticias/qa_sports.json")
