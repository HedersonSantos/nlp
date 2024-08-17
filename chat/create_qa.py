import pandas as pd
import os, json
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, BertForQuestionAnswering, BertTokenizer, pipeline


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

def load_or_save_model_mt5(model_save_path):
    mt5_model_name = "google/mt5-small"
    try:
        model_mt5, tokenizer_mt5 = load_model(MT5ForConditionalGeneration, MT5Tokenizer, mt5_model_name, model_save_path)
        return model_mt5, tokenizer_mt5
    except OSError:
        model_mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_model_name)
        tokenizer_mt5 = MT5Tokenizer.from_pretrained(mt5_model_name)
        save_model(model_mt5, tokenizer_mt5, mt5_model_name, model_save_path)
        return model_mt5, tokenizer_mt5

def load_or_save_model_bertimbau(model_save_path):
    bertimbau_model_name = "neuralmind/bert-base-portuguese-cased"
    try:
        model_bertimbau, tokenizer_bertimbau = load_model(BertForQuestionAnswering, BertTokenizer, bertimbau_model_name, model_save_path)
        return model_bertimbau, tokenizer_bertimbau
    except OSError:
        model_bertimbau = BertForQuestionAnswering.from_pretrained(bertimbau_model_name)
        tokenizer_bertimbau = BertTokenizer.from_pretrained(bertimbau_model_name)
        save_model(model_bertimbau, tokenizer_bertimbau, bertimbau_model_name, model_save_path)
        return model_bertimbau, tokenizer_bertimbau
        
        
def generate_question(text, max_new_tokens=200):
    # Formatar texto para geração de perguntas com mT5
    input_text = f"gerar pergunta: {text}"
    print('tokenizar texto')
    input_ids = tokenizer_mt5.encode(input_text, return_tensors="pt")
    print(f'tokens:{input_ids}')
    print('submeter tokens ao modelo')      
    outputs = model_mt5.generate(input_ids, max_new_tokens=max_new_tokens)
    print(f'pergunta: {outputs}')      
    #print('tokenizar pergunta')
    question = tokenizer_mt5.decode(outputs[0], skip_special_tokens=True)
    print(f'question: {question}')
    return outputs

    # Gerar pergunta
    outputs = model_mt5.generate(input_ids)
    question = tokenizer_mt5.decode(outputs[0], skip_special_tokens=True)
    return question

def generate_answer(question, context):
    # Gerar resposta usando BERTimbau
    response = qa_pipeline(question=question, context=context)
    return response['answer']

def read_parquet_file(directory_path):
    parquet_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".parquet")]
    combined_df = pd.concat((pd.read_parquet(f) for f in parquet_files), ignore_index=True)
    return combined_df

def process_texts():
    data = []
    df_textos =  read_parquet_file('/home/jupyter/textos/dataset_noticias')
    for index, row in df_textos.iterrows():
        #text=row['text']
        text="Pela primeira vez desde que assumiu o comando da seleção brasileira há quase 16 meses, o técnico Tite afirmou que já pensou em continuar no cargo após a Copa.  Ele assumiu o posto em junho de 2016 após a demissão de Dunga.  O treinador iniciou o seu trabalho fora da zona de classificação para a Copa do Mundo, mas com uma arrancada surpreendente conseguiu nove vitórias consecutivas e dois empates e obteve a vaga no Mundial com antecipação. O treinador ainda coleciona mais duas vitórias e uma derrota em jogos amistosos."
        question=generate_question(text)
        print(question)
        print('**********************')
        answer = generate_answer(question, text)
        print(answer)
        data.append({"instruction": question, "response": answer})
        break
    return data


#carrega os modelos envolvidos
model_mt5, tokenizer_mt5 = load_or_save_model_mt5('/home/jupyter/textos/modelos_qa/mt5')
model_bertimbau, tokenizer_bertimbau = load_or_save_model_bertimbau('/home/jupyter/textos/modelos_qa/bertimbau')
qa_pipeline = pipeline("question-answering", model=model_bertimbau, tokenizer=tokenizer_bertimbau)

# Processar textos e gerar perguntas e respostas
qa_pairs = process_texts()

# Salvar resultado em um arquivo JSONL
#with open('qa_pairs.jsonl', 'w', encoding='utf-8') as f:
#    for pair in qa_pairs:
#        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

#print("Geração de perguntas e respostas concluída e salva em 'qa_pairs.jsonl'.")
