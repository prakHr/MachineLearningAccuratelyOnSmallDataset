from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import qnn_circuit
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import StatevectorSampler as Sampler
import pandas as pd
import textfeatures as tf




# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


def binary_classification_specially_accurate_on_small_samples_for_tabular_dataset(to_predict_label = "setosa",file_path='iris.csv', target_column='species',maxiter=100):
    sampler = Sampler()
    data = pd.read_csv(file_path)
    target_column = target_column
    X = data.drop(target_column, axis=1).values
    y = data[target_column].values
    print(X)

    num_features = X.shape[1]
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # small_sample_size = 20
    # X = X[:small_sample_size]
    # y = y[:small_sample_size]
    X = MinMaxScaler().fit_transform(X)
    y_cat = []
    for label in y:
        if label == to_predict_label:
            y_cat.append(to_predict_label)
        else:
            y_cat.append(f"Not that predicted label")
            
    y_cat = np.array(y_cat)

    vqc = VQC(
        num_qubits=num_features,
        optimizer=COBYLA(maxiter=maxiter),
        sampler=sampler,
    )

    # fit classifier to data
    vqc.fit(X, y_cat)

    # score classifier
    vqc.score(X, y_cat)

    predict = vqc.predict(X)
    # print(f"Predicted:, Ground truth:")
    score = 0
    l = len(predict)
    for a,b in zip(predict, y_cat):
        if a == b:
            score += 1
    #     print(f"{a}, {b}")
    return vqc,score/l


def binary_classification_specially_accurate_on_small_samples_for_text_dataset(text_column='v2', to_predict_label = "spam", file_path='text.csv', target_column='v1', maxiter=100):
    df = pd.read_csv(file_path, encoding="latin1")
    features = ["word_cnt", "char_len", "avg_wrd_length", "stopwords_cnt"]
    
    tf.word_count(df,text_column,"word_cnt")
    tf.char_count(df,text_column,"char_len")
    tf.avg_word_length(df,text_column,"avg_wrd_length")
    tf.stopwords_count(df,text_column,"stopwords_cnt")
    tf.stopwords(df,text_column,"stopwords")
    
    X = df[features].values
    y = df[target_column].values

    sampler = Sampler()
    num_features = X.shape[1]
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    X = MinMaxScaler().fit_transform(X)
    y_cat = []
    for label in y:
        if label == to_predict_label:
            y_cat.append(to_predict_label)
        else:
            y_cat.append(f"Not that predicted label")
            
    y_cat = np.array(y_cat)

    vqc = VQC(
        num_qubits=num_features,
        optimizer=COBYLA(maxiter=maxiter),
        sampler=sampler,
    )

    # fit classifier to data
    vqc.fit(X, y_cat)

    # score classifier
    vqc.score(X, y_cat)

    predict = vqc.predict(X)
    score = 0
    l = len(predict)
    for a,b in zip(predict, y_cat):
        if a == b:
            score += 1
    return vqc,score/l

# if __name__ == "__main__":
    # file_path='iris.csv'
    # target_column='species'
    # to_predict_label = "setosa"
    # maxiter = 200
    # model, score = binary_classification_specially_accurate_on_small_samples_for_tabular_dataset(to_predict_label=to_predict_label, file_path=file_path, target_column=target_column, maxiter=maxiter)
    # print(f"Model Score: {score}")
    # print(f"Model: {model}")

    # file_path='text.csv'
    # target_column='v1'      
    # text_column='v2'
    # to_predict_label = "spam"
    # maxiter = 200
    # model, score = binary_classification_specially_accurate_on_small_samples_for_text_dataset(text_column=text_column, to_predict_label=to_predict_label, file_path=file_path, target_column=target_column, maxiter=maxiter)
    # print(f"Model Score: {score}")
    # print(f"Model: {model}")