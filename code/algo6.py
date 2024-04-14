from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
def algo6(file):
    # Load SBERT model
    model = SentenceTransformer('all-mpnet-base-v2')
    df = pd.read_csv(f'{file}_data.csv')
    df = df.drop(columns = ['comments'],axis=1)
    df.dropna(inplace=True)
    # Example sentences
    df.dropna(subset=['method_name'],inplace=True)
    sentences = []
    print(df.shape)
    returns = df['returns'].tolist()
    method_name = df['method_name'].tolist()
    sentences = []
    for(i,j) in zip(returns,method_name):
        sentences.append([i,j])
    print(len(sentences),len(sentences[0]))
    # Encode sentences to get the embeddings
    emmd = []
    for i in tqdm(range(len(sentences))):
        embeddings_i = model.encode(sentences[i][0])
        embeddings_j = model.encode(sentences[i][1])
        emmd.append([embeddings_i,embeddings_j])
    res = []
    theta = 0.245
    for i in range(len(emmd)):
        similarity_matrix = cosine_similarity(emmd[i])
        if similarity_matrix[0][1] < theta:
            res.append('negative')
        else:
            res.append('positive')

    df['label'] = res
    df.to_csv(f'{file}_algo6_data.csv',index=False)
    
    
if __name__ == "__main__":
    file = input('Enter the file name: ')
    algo6(file)