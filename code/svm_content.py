from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import numpy as np
def svm(project='buildship'):
    df = pd.read_csv(f'../data/{project}_combined.csv')
    df = df.dropna(subset=['content'])
    df.drop(columns=['access_modifier','return_type','method_name','parameters','returns','comments'],inplace=True)
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    X = tfidf_vectorizer.fit_transform(df['content'])
    y = df['catagory']
    # y = label_binarize(y, classes=np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_clf = SVC(kernel='rbf', C=1.0,gamma=0.09999999999999999)  # You can adjust the kernel and C parameter as needed
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['get','not_LA','is','set']))
    #     # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(len(['get','not_LA','is','set'])):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # # Plot ROC curve for each class
    # plt.figure(figsize=(10, 8))
    # for i in range(len(['get','not_LA','is','set'])):
    #     plt.plot(fpr[i], tpr[i], label='ROC curve for class %s (AUC = %0.2f)' % (['get','not_LA','is','set'][i], roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve for Multiclass AdaBoost Classification')
    # plt.legend(loc="lower right")
    # plt.show()
if __name__ == '__main__':
    svm()