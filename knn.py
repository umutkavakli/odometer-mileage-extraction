import numpy as np
import matplotlib.pyplot as plt
from utils import load_classification_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == '__main__':
    accuracy = []
    precision = []
    recall = []
    f1 = []

    trainX, trainY, testX, testY = load_classification_data()
    
    for i in range(5, 80):
        print(i)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(trainX, trainY)
        preds = knn.predict(testX)
        
        accuracy.append(accuracy_score(preds, testY))
        precision.append(precision_score(preds, testY))
        recall.append(recall_score(preds, testY))
        f1.append(f1_score(preds, testY))

    for metric in [(accuracy, 'accuracy'), (precision, 'precision'), (recall, 'recall'), (f1, 'f1')]:
        plt.figure(figsize=(10,10))
        plt.plot(range(5,80), metric[0], color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title(f'{metric[1]} vs K value')
        plt.xlabel('K')
        plt.ylabel(f'{metric[1]}')
        plt.savefig(f'knn - {metric[1]}')

    print(max(accuracy), 5 + np.argmax(accuracy))
    print(max(accuracy), 5 + np.argmax(accuracy))
    print(max(accuracy), 5 + np.argmax(accuracy))
    print(max(accuracy), 5 + np.argmax(accuracy))    
