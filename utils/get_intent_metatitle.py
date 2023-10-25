from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import numpy as np
import json
from utils.client import generate_seo_metatitle



# id2label= {0: 'Commercial',
#  1: 'Informational',
#  2: 'Navigational',
#  3: 'Local',
#  4: 'Transactional'}

# label2id= {'Commercial': 0,
#  'Informational': 1,
#  'Navigational': 2,
#  'Local': 3,
#  'Transactional': 4}


# removed local
id2label= {0: 'Commercial', 1: 'Informational', 2: 'Navigational', 3: 'Transactional'}
label2id= {'Commercial': 0, 'Informational': 1, 'Navigational': 2, 'Transactional': 3}

model_name= "intent_classification_model_with_metatitle_with_local2/checkpoint-2700"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")


# probabilities = 1 / (1 + np.exp(-logit_score))
def logit2prob(logit):
    # odds =np.exp(logit)
    # prob = odds / (1 + odds)
    prob= 1/(1+ np.exp(-logit))
    return np.round(prob, 3)





def get_intent_one_by_one(keyword:str):
    inputs = tokenizer(generate_seo_metatitle(keyword), padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # print("logits: ", logits)
    # predicted_class_id = logits.argmax().item()
    
    # get probabilities using softmax from logit score and convert it to numpy array
    # probabilities_scores = F.softmax(logits.cpu(), dim = -1).numpy()[0]
    individual_probabilities_scores = logit2prob(logits.cpu().numpy()[0])
    
    score_list= []
    
    for i in range(len(id2label)):
        label= id2label[i]
        
        score= individual_probabilities_scores[i]
        if score>0.5:
            score_list.append(
                        (label, score)
                    )
        # if score>=0.5: 
        #     score_list.append(
        #         (id2label[i], score)
        #     )
            
    if len(score_list)==0:
        score_list.append(("undefined",1))        
    score_list.sort(
        key= lambda x: x[1], reverse=True
    )
            
    return score_list
    # return (np.argmax(i), id2label[np.argmax(i)])
    



def get_intent_one_by_one_test(metatitle:str):
    inputs = tokenizer(metatitle,padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # print("logits: ", logits)
    # predicted_class_id = logits.argmax().item()
    
    # get probabilities using softmax from logit score and convert it to numpy array
    # probabilities_scores = F.softmax(logits.cpu(), dim = -1).numpy()[0]
    individual_probabilities_scores = logit2prob(logits.cpu().numpy()[0])
    
    score_list= []
    
    for i in range(len(id2label)):
        label= id2label[i]
        
        score= individual_probabilities_scores[i]
        if score>0.5:
            score_list.append(
                        (label, score)
                    )
        # if score>=0.5: 
        #     score_list.append(
        #         (id2label[i], score)
        #     )
            
    if len(score_list)==0:
        score_list.append(("undefined",1))        
    score_list.sort(
        key= lambda x: x[1], reverse=True
    )
            
    return score_list
    # return (np.argmax(i), id2label[np.argmax(i)])
    
    

