import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
df = pd.read_csv('is_labeled_data.csv')
df = df.drop(columns=["comments"],axis=1)
df = df.dropna()
# access_modifier,return_type,method_name,parameters,returns,comments,content,label

df['combined'] = df['access_modifier']+' '+df['return_type']+' '+df['method_name']+'('+df['parameters']+')'+df['content']
X = df['combined'].tolist()
y = df['label'].map({'positive': 1, 'negative': 0}).to_list()
# Manually split your dataset into training and validation sets if not already done

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# MODEL_NAME = 'microsoft/codebert-base'
# t = text.Transformer(MODEL_NAME, maxlen=512, class_names=['negative', 'positive'])
# # Note: Adjust maxlen according to your data's characteristics
# trn = text.Transformer(MODEL_NAME, maxlen=512, class_names=['0', '1'])
# train_data = trn.preprocess_train(X_train, y_train)
# val_data = trn.preprocess_test(X_val, y_val)


# # Get classifier model
# model = trn.get_classifier()

# # Create a Learner object
# learner = ktrain.get_learner(model, train_data=train_data, val_data=val_data, batch_size=6)

# # Train the model
# learner.fit_onecycle(5e-5, 1)  # Adjust epochs and learning rate as needed

# print(learner.validate(class_names=t.get_classes()))


# predictor = ktrain.get_predictor(learner.model, preproc=t)
predictor = ktrain.load_predictor('./code_bert_model')
print(predictor.predict('public void setupRegistry() { stateBeforeInjectorCreation = GlobalRegistries.makeCopyOfGlobalState(); if (injector == null) { getInjector(); } stateAfterInjectorCreation.restoreGlobalState(); }'))

predictor.explain('public void setupRegistry() { stateBeforeInjectorCreation = GlobalRegistries.makeCopyOfGlobalState(); if (injector == null) { getInjector(); } stateAfterInjectorCreation.restoreGlobalState(); }')