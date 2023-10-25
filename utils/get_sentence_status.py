from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import numpy as np
import re

tokenizer = AutoTokenizer.from_pretrained("gpt3_finetuned_model/checkpoint-30048")
tokenizer_v2 = AutoTokenizer.from_pretrained("gpt2-large")


model = AutoModelForSequenceClassification.from_pretrained("gpt3_finetuned_model/checkpoint-30048").to("cuda")


# probabilities = 1 / (1 + np.exp(-logit_score))
def logit2prob(logit):
    # odds =np.exp(logit)
    # prob = odds / (1 + odds)
    prob= 1/(1+ np.exp(-logit))
    return np.round(prob, 3)

def split_sentence(sentence:str):
    # Create a regular expression pattern from the list of separators
    sentence= sentence.replace('\n', '')
    separators = ['. ', '.', ':']
    
    pattern = '|'.join(map(re.escape, separators))

    # Split the sentence using the pattern as a delimiter
    parts = re.split(pattern, sentence)

    return parts



def predict(sentence: str):
    '''
    Returns (probability_human, probability_AI, label)
    '''
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
        
    print("logits: ", logits)
    predicted_class_id = logits.argmax().item()
    # get probabilities using softmax from logit score and convert it to numpy array
    probabilities_scores = np.round(
                                F.softmax(logits.to("cpu"), dim = -1).numpy()[0],
                                3)
    print("P(Human): ", probabilities_scores[0])
    print("P(AI): ", probabilities_scores[1])
    label= "Human Written" if model.config.id2label[predicted_class_id]=='NEGATIVE' else 'AI written'
    print("Label: ", label)
    print(model.config.id2label[predicted_class_id])
    
    
    return probabilities_scores[0], probabilities_scores[1], label
    





def calculate_burstiness(sentence: str):
    '''
    Returns (variance, average_length)
    '''
    list_of_sentences= split_sentence(sentence)
    arr= []
    for i in list_of_sentences:
        if len(i)==0:
            continue
        ei= tokenizer_v2(i, return_tensors="pt")
        arr.append(ei.input_ids.size(1))
        
    variance= np.var(np.array(arr))
    std_deviation= np.std(np.array(arr))
    avg_length= np.average(np.array(arr))
    
    print(f"arr= {(arr)}")
    print(f'variance: {variance}')
    print(f'std: {std_deviation}')
    print(f'average length: {avg_length}')
    
    return variance,  avg_length
    
    
    
    
    
def complete_sentence_analysis(sentence:str):
    '''
    Returns a dictionary 
    {
        p_human : probablity that the text is written by the human
        p_ai : probablity that the text is written by ai
        label : label {ai/human}
        variance : variance in the length of the sentences
        avg_length: average tokens per sentence
    }
    '''
    p_human, p_ai, label= predict(sentence)
    variance, avg_length= calculate_burstiness(sentence)
    return {
        "p_human": p_human,
        "p_ai": p_ai,
        "label": label,
        "variance": variance,
        "avg_length": avg_length
    }
    
    
    
    
    
def get_top_labels(keyword: str):
    '''
    Returns score list
    '''
    inputs = tokenizer(keyword, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # print("logits: ", logits)
    # predicted_class_id = logits.argmax().item()
    
    # get probabilities using softmax from logit score and convert it to numpy array
    # probabilities_scores = F.softmax(logits.cpu(), dim = -1).numpy()[0]
    individual_probabilities_scores = logit2prob(logits.cpu().numpy()[0])
    
    score_list= []
    
    for i in range(2):
        label= "Human Written" if model.config.id2label[i]=='NEGATIVE' else 'AI written'
        
        score= individual_probabilities_scores[i]
        score_list.append(
                    (label, score)
                )
        # if score>=0.5: 
        #     score_list.append(
        #         (id2label[i], score)
        #     )
            
            
    score_list.sort(
        key= lambda x: x[1], reverse=True
    )
            
    return score_list[:5]