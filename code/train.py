

# import ktrain
# import pandas as pd
# from ktrain import tabular
# import numpy as np
# import pandas   as pd

# from sklearn.metrics import matthews_corrcoef, roc_auc_score


# def get_model(file):
#     # load the model
#     df = pd.read_csv(file)
#     df['label'] = df['label'].replace({'positive': 1, 'negative': 0})
#     print(f"the replced values : {df['label'].value_counts()}")
#     df['index'] = df.index
#     np.random.seed(42)
#     p = 0.1 # 10% for test set
#     prop = 1-p
#     temp_df = df.copy()
#     msk = np.random.rand(len(temp_df)) < prop
#     train_df = temp_df[msk]
#     test_df = temp_df[~msk]
#     print(train_df["label"].value_counts())
#     print(test_df["label"].value_counts())
#     trn,val,preproc = tabular.tabular_from_df(train_df, label_columns=['label'], random_state=42)
#     model  = tabular.tabular_classifier(name='mlp',train_data=trn)
#     learner = ktrain.get_learner(model,train_data=trn,val_data=val,batch_size=6)
#     learner.fit_onecycle(5e-3, 1)
#     print(learner.evaluate(val, class_names=preproc.get_classes()))
#     predictor = ktrain.get_predictor(learner.model, preproc)
#     preds = predictor.predict(test_df, return_proba=True)
#     # Calculate AUC
#     # Assuming the positive class is at index 1
#     auc_score = roc_auc_score(test_df['label'].values, preds[:, 1])
#     print('AUC:', auc_score)
#     # Calculate MCC
#     predicted_labels = np.argmax(preds, axis=1)
#     mcc_score = matthews_corrcoef(test_df['label'].values, predicted_labels)
#     print('MCC:', mcc_score)
#     predictor.explain(test_df,row_num=2,class_id=1)
#     model_path = f'model{file.split(".")[-2]}'
#     print(model_path)
#     predictor.save(model_path)
#     reloaded_predictor = ktrain.load_predictor(f'model{file.split(".")[-2]}')
#     reloaded_predictor.predict(test_df)[:5]
    
# # df = df.drop(columns=["comments"],axis=1)

# # change the label data by replacing negative with 0 and positive with 1
# # df["label"] = df["label"].replace("negative",0).replace("positive",1)
# if __name__ == "__main__":
#     file = input('Enter the file name: ')
#     get_model(file)

import ktrain
import pandas as pd
from ktrain import tabular
import numpy as np
import pandas   as pd
import json

from sklearn.metrics import matthews_corrcoef, roc_auc_score


def get_model(file):
    # load the model
    df = pd.read_csv(file)
    df = df.sample(frac=1).reset_index(drop=True)
    df['label'] = df['label'].replace({'positive': 1, 'negative': 0})
    # df['code'] = df
    print(f"the replced values : {df['label'].value_counts()}")
    df['index'] = df.index
    np.random.seed(42)
    p = 0.1 # 10% for test set
    prop = 1-p
    temp_df = df.copy()
    msk = np.random.rand(len(temp_df)) < prop
    train_df = temp_df[msk]
    test_df = temp_df[~msk]
    print(train_df["label"].value_counts())
    print(test_df["label"].value_counts())
    return train_df, test_df


def get_lean(train_df,test_df):
    trn,val,preproc = tabular.tabular_from_df(train_df, label_columns=['label'], random_state=42)
    model  = tabular.tabular_classifier(name='mlp',train_data=trn)
    learner = ktrain.get_learner(model,train_data=trn,val_data=val,batch_size=6)
    learner.fit_onecycle(5e-3, 1)
    print(learner.evaluate(val, class_names=preproc.get_classes()))
    predictor = ktrain.get_predictor(learner.model, preproc)
    preds = predictor.predict(test_df, return_proba=True)
    try:
        auc_score = roc_auc_score(test_df['label'].values, preds[:, 1])
        print('AUC:', auc_score)
        # Calculate MCC
        predicted_labels = np.argmax(preds, axis=1)
        mcc_score = matthews_corrcoef(test_df['label'].values, predicted_labels)
        print('MCC:', mcc_score)
        predictor.explain(test_df,row_num=2,class_id=1)
    except ValueError:
        print('ValueError')
    
    return predictor


def model_save(file,predictor,test_df):
    model_path = f'{file.split(".")[-2]}'
    print()
    print()
    model_path = f'../model/{model_path.split("/")[-1]}'
    print(model_path)
    print()
    print()
    predictor.save(model_path)
    reloaded_predictor = ktrain.load_predictor(model_path)
    reloaded_predictor.predict(test_df)[:5]
    
    
    
    
def train_and_save_model(list_of_processed_files):
    for file in list_of_processed_files:
        train_df, test_df = get_model(file)
        predictor = get_lean(train_df,test_df)
        model_save(file,predictor,test_df)
        
        

def train_files():
    data_files = {'is':['is_labeled.csv', 'is_labeled_output.csv'], 'set':['set_labeled.csv', 'set_labeled_output.csv'],'get':['get_labeled.csv', 'get_labeled_output.csv'],'algo5':['algo5_data.csv'],'algo6':['algo6_data.csv']}
    list_of_data = ['is', 'set', 'get', 'algo5', 'algo6']
    file = 'projects_list.json'
    list_of_processed_files = []
    try:
        with open(file) as f:
            projects = json.load(f)
            for project in projects:
                # Load the two CSV files into separate DataFrames
                for i in range(1,3):
                    try:
                        df1 = pd.read_csv(f'../data/{project}_{data_files[list_of_data[i]][0]}')
                        df2 = pd.read_csv(f'../data/{project}_{data_files[list_of_data[i]][1]}')
                        # print(list_of_data[i],df1['label'].value_counts())
                    #     # Concatenate the two DataFrames
                        df1 = df1[df1['label'] == 'negative']
                        combined_df = pd.concat([df1, df2])
                        print('combined_df')
                        # print(combined_df['label'].value_counts())
                    #     # # Remove duplicate rows
                    #     # combined_df = combined_df.drop_duplicates()
                        # Write the combined DataFrame to a new CSV file
                        combined_file = f'../data/{project}_{list_of_data[i]}_combined.csv'
                        combined_df.to_csv(combined_file, index=False)
                        list_of_processed_files.append(combined_file)
                        # print(combined_file.split('.')[-2])
                        # train.get_model(combined_file)
                    except FileNotFoundError:
                        print(f"File not found: {project}_{data_files[list_of_data[i]][0]} or {project}_{data_files[list_of_data[i]][1]}")
                for i in range(3,5):
                    try:
                        df = pd.read_csv(f'../data/{project}_{data_files[list_of_data[i]][0]}')
                        # print(list_of_data[i],df['label'].value_counts())
                        # df = df[df['label'] == 'negative']
                        list_of_processed_files.append(f'../data/{project}_{data_files[list_of_data[i]][0]}')
                    except FileNotFoundError:
                        print(f"File not found: {project}_{data_files[list_of_data[i]][0]}")
    except FileNotFoundError:
        print(f"File not found: {file}")
        
    return list_of_processed_files





if __name__ == "__main__":
    list_of_processed_files = train_files()
    print(list_of_processed_files)
    train_and_save_model(list_of_processed_files)
    print(list_of_processed_files)
    # file = input('Enter the file name: ')
    # get_model(file)
    
    
    
