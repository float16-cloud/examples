from transformers import AutoTokenizer, AutoModel
import torch

model_path = '../../model-weight/bge-m3'
content = ["สวัสดี","ผัดไท","food","car"]
embedding_tokenizer = AutoTokenizer.from_pretrained(model_path)
embedding_model = AutoModel.from_pretrained(model_path).half().to('cuda')

try :
    content_length_array = []
    for sentence in content : 
        content_tokenized = embedding_tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        content_length = content_tokenized['input_ids'].shape[1]
        content_length_array.append(content_length)

    model_input = embedding_tokenizer(content, padding=True, truncation=True, return_tensors='pt').to('cuda:0')

    with torch.no_grad():
        model_output = embedding_model(**model_input)
        sentence_embeddings = model_output[0][:, 0]

    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    sentence_list = sentence_embeddings.tolist()
    
    print(sentence_list)
    
    for idx,sentence in enumerate(sentence_list) : 
        with open(f'output_{idx}.txt','w') as f : 
            f.write(f"{content[idx]},{str(sentence)}")

except Exception as e : 
    with open(f'error.txt','w') as f : 
        f.write(str(e))