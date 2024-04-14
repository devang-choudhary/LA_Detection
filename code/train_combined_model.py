import ktrain
import pandas as pd
from ktrain import tabular
import numpy as np
import pandas   as pd
from sklearn.utils import shuffle
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



def get_model(a,project):
    df = pd.read_csv(f'../data/{project}_combined.csv')
    df = df.drop(df.index)
    print(df.shape)
    for p,u in a.items():
        if p == project:
            continue
        df1 = pd.read_csv(f'../data/{p}_combined.csv')
        df = pd.concat([df,df1])
    df = df.drop(columns=['comments'],axis=1)
    train_df = shuffle(df)
    test_df = pd.read_csv(f'../data/{project}_combined.csv')
    test_df = test_df.drop(columns=['comments'],axis=1)
    test_df['index'] = test_df.index
    train_df['index'] = train_df.index  
    return train_df, test_df

def get_lean(train_df,test_df):
    print(test_df.iloc[1])
    trn,val,preproc = tabular.tabular_from_df(train_df, label_columns=['catagory'], random_state=42)
    model  = tabular.tabular_classifier(name='mlp',train_data=trn)
    learner = ktrain.get_learner(model,train_data=trn,val_data=val,batch_size=32)
    # learner.lr_find(show_plot=True, max_epochs=5)
    data = {"get":0,"is":1,"not_LA":2,"set":3}
    learner.fit_onecycle(5e-3, 1)
    print(learner.evaluate(val, class_names=preproc.get_classes()))
    predictor = ktrain.get_predictor(learner.model, preproc)    
    return predictor


def model_save(predictor,project):
    model_path = f"model/{project}_combined_model"
    predictor.save(model_path)
    
    
a = {
    "buildship": " ", 
    "eclips-collections": " ",
    "jifa": " ",
    "jkube": " ",
    "hawkbit": "https://github.com/eclipse/hawkbit.git",
    "kura": "https://github.com/eclipse/kura.git",
    "milo": "https://github.com/eclipse/milo.git",
    "openvsx": "https://github.com/eclipse/openvsx.git",
    "steady": "https://github.com/eclipse/steady.git",
    "xtext": "https://github.com/eclipse/xtext.git"
}



def train(project):
    train_df, test_df = get_model(a,project)
    predictor = get_lean(train_df,test_df)
    model_save(predictor,project)
    print("Model trained and saved successfully")
    
if __name__ == "__main__":
    maping = {}
    count = 0
    for project,_ in a.items():
        count+=1
        print(f"{project} -> {count}")
        maping[count] = project
    project = maping[int(input("Enter the project number: "))]
    train(project)