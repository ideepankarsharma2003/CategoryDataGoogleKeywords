from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import numpy as np
import json



label2id= json.load(
    open('data/categories_refined.json', 'r')
)
id2label= {}
for key in label2id.keys():
    id2label[label2id[key]] = key
 


model_name= "intent_classification_model/checkpoint-1216"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")


# probabilities = 1 / (1 + np.exp(-logit_score))
def logit2prob(logit):
    # odds =np.exp(logit)
    # prob = odds / (1 + odds)
    prob= 1/(1+ np.exp(-logit))
    return np.round(prob, 3)



  
def get_top_intent(keyword: str):
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
    
    for i in range(5):
        label= model.config.id2label[i]
        
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
            
    return score_list