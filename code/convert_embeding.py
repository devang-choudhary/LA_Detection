import pandas as pd
import numpy as np
# train_df = pd.read_csv("combined/buildship_combined.csv")
test_df = pd.read_csv("../data/xtext_combined.csv")
from transformers import AutoTokenizer, AutoModel
test_df1 = test_df[:10]
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
count = 0
def encode_content(content):
    # count+=1
    # print(count)
    inputs = tokenizer(content, return_tensors="pt", max_length=100, truncation=True, padding="max_length")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].detach().numpy()

# Apply this function to your content in the train and test set
# This step is simplified; in practice, you might batch process this for efficiency
# train_df['content_embedding'] = train_df['content'].apply(encode_content)
test_df['content_embedding'] = test_df['content'].apply(encode_content)
# train_df.to_csv('buildship_combined_altered.csv',index = False)
test_df.to_csv('../data/xtext_combined_altered.csv',index = False)