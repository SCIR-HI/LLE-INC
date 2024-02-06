# -*- coding: utf-8 -*-
# author:Haochun Wang
import argparse
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding, LocallyLinearEmbeddingMod


num_training = {'SST2': 32, 'MRPC': 32, 'QNLI': 32, 'QQP': 32, 'RTE': 32, 'MNLI': 48}
num_label = {'SST2': 2, 'MRPC': 2, 'QNLI': 2, 'QQP': 2, 'RTE': 2, 'MNLI': 3}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--seed')
    parser.add_argument('--dataset_name')
    parser.add_argument('--log_dir')
    parser.add_argument('--manifold')
    parser.add_argument('--res_dir')
    parser.add_argument('--kn')
    parser.add_argument('--split')
    parser.add_argument('--dim')
    parser.add_argument('--dim_split')
    args = parser.parse_args()

    return args

def read_pickle(m='train'):
    x = []
    y = []
    if m == 'train':
        folder_dir = '/train_rep/'
        for i in range(0, num_training[args.dataset_name]):
            with open(os.path.join(folder_dir, args.dataset_name, '%s-%s.pickle' %(str(args.seed), str(i))), 'rb') as f:
                a = pickle.load(f)
                x.append(a[1])
                y.append(a[0][1])
    else:
        folder_dir = '/test_rep/'
        if m == 'train-all':
            folder_dir = '/train_rep_all/'
        files_ls = os.listdir(os.path.join(folder_dir, args.dataset_name))
        for i in files_ls:
            if 'pickle' in i:
                try:
                    with open(os.path.join(folder_dir, args.dataset_name, i), 'rb') as f:
                        a = pickle.load(f)
                        x.append(a[1])
                        y.append(a[0][1])
                except EOFError:
                    continue

    return x, y

def word2label(label_list):
    label_index = 0
    word2label_dic = {}
    for i in label_list:
        if i not in word2label_dic:
            word2label_dic[i] = label_index
            label_index += 1
    return word2label_dic

def knn(training_set, test_set):
    knn_X, knn_y = training_set
    w2l_dic = word2label(knn_y)
    knn_y = [w2l_dic[i] for i in knn_y]

    test_X, test_y = test_set
    test_y = [w2l_dic[i] for i in test_y]

    for k_n in range(1, int(args.kn)+1, int(args.split)):
        mani_model = KNeighborsClassifier(n_neighbors=k_n, metric='cosine', n_jobs=-1)
        mani_model.fit(knn_X, knn_y)
        if args.dataset_name in ['QQP', 'MRPC']:
            a = f1_score(test_y, mani_model.predict(test_X))
        else:
            a = accuracy_score(test_y, mani_model.predict(test_X))
        print(f"k_n:{k_n}, Test_acc:{a}")
        with open(os.path.join(args.res_dir, '%s_knn.txt') % args.dataset_name,'a') as fn:
            fn.write(f"k_n:{k_n}, Test_acc:{a}\n")

def lle(training_set, test_set):
    knn_X, knn_y = training_set
    w2l_dic = word2label(knn_y)
    knn_y = [w2l_dic[i] for i in knn_y]

    test_X, test_y = test_set
    test_y = [w2l_dic[i] for i in test_y]
    for k_n in range(1, int(args.kn)+1, int(args.split)):
        for dim in range(1, int(args.dim)+1, int(args.dim_split)):
            try:
                mani_model = LocallyLinearEmbedding(n_components=dim, n_neighbors=k_n,
                                                    n_jobs=30,
                                                    eigen_solver='dense')
                transformed_X = mani_model.fit_transform(knn_X)

                tran_test_X = mani_model.transform(test_X)
                mani_model = KNeighborsClassifier(n_neighbors=k_n)
                mani_model.fit(transformed_X.tolist(), knn_y)
                test_X_ = tran_test_X.tolist()
                if args.dataset_name in ['QQP', 'MRPC']:
                    a = f1_score(test_y, mani_model.predict(test_X_))
                else:
                    a = accuracy_score(test_y, mani_model.predict(test_X_))
                print(f"k_n:{k_n}, dim:{dim}, Test_acc:{a}")
                with open(os.path.join(args.res_dir, '%s_lle.txt') % args.dataset_name, 'a') as fn:
                    fn.write(f"k_n:{k_n}, dim:{dim}, Test_acc:{a}\n")
            except ValueError:
                print(f"k_n:{k_n}, dim:{dim}, Failed lle.")
                break


def lle_mod(training_set, test_set):
    knn_X, knn_y = training_set
    w2l_dic = word2label(knn_y)
    knn_y = [w2l_dic[i] for i in knn_y]

    test_X, test_y = test_set
    test_y = [w2l_dic[i] for i in test_y]
    # for k_n in range(1, int(args.shot) + 1):
    for k_n in range(1, int(args.kn)+1, int(args.split)):
        for dim in range(1, int(args.dim)+1, int(args.dim_split)):
            try:
                mani_model = LocallyLinearEmbeddingMod(n_components=dim, n_neighbors=k_n,
                                                       n_jobs=30,
                                                       eigen_solver='dense',
                                                       X_label=knn_y)
                transformed_X = mani_model.fit_transform(knn_X)
                tran_test_X = mani_model.transform(test_X)
                mani_model = KNeighborsClassifier(n_neighbors=k_n)
                mani_model.fit(transformed_X.tolist(), knn_y)
                test_X_ = tran_test_X.tolist()
                if args.dataset_name in ['QQP', 'MRPC']:
                    a = f1_score(test_y, mani_model.predict(test_X_))
                else:
                    a = accuracy_score(test_y, mani_model.predict(test_X_))
                print(f"k_n:{k_n}, dim:{dim}, Test_acc:{a}")
                with open(os.path.join(args.res_dir, '%s_lle_mod.txt') % args.dataset_name, 'a') as fn:
                    fn.write(f"k_n:{k_n}, dim:{dim}, Test_acc:{a}\n")
            except ValueError:
                print(f"k_n:{k_n}, dim:{dim}, Failed lle-mod.")
                break

args = parse_args()

training_set = read_pickle(m='train-all')
test_set = read_pickle(m='test')
if args.manifold == 'knn':
    knn(training_set, test_set)
elif args.manifold == 'lle':
    lle(training_set, test_set)
elif args.manifold == 'lle-mod':
    lle_mod(training_set, test_set)
else:
    print('error')
