from keys import endpoint, key
# endpoint = "https://eastus.api.cognitive.microsoft.com/"

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Authenticate the client using your key and endpoint 
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()


def replace_original_text_with_entities(original_text:str):
    try:
        result = client.recognize_entities(documents = [original_text])[0]

        for entity in result.entities:
            # print("\tText: \t", entity.text, "\tCategory: \t", entity.category, "\tSubCategory: \t", entity.subcategory,
            #         "\n\tConfidence Score: \t", round(entity.confidence_score, 2), "\tLength: \t", entity.length, "\tOffset: \t", entity.offset, "\n")
            original_text= original_text.replace(
                entity.text, 
                entity.text+ f' ({entity.category}) '
            )
        return original_text

    except Exception as err:
        
        print("Encountered exception. {}".format(err))
        return original_text
    