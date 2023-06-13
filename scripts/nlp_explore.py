import numpy as np
from time import time
import ot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import multivariate_normal
import seaborn as sn
import pandas as pd
from pathlib import Path
from sklearn.utils import check_random_state
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import pickle
# User functions
# from utils import BaseTransport, GaussianTransport, get_trajectories_for_plotting, calc_parsimony, W2_dist
import os, sys
import argparse

import time as Time

import json
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch.nn.functional as F

from src.optimal_transport import group_feature_transport, transform_samples, group_mean_shift_transport, group_kmeans_shift_transport, transform_samples_kmeans

from src.training import regular_training, dro_training
from src.logistic_regression import PTLogisticRegression, PTNN, FFNetwork
from src.training import regular_training
from src.distance import group_percent_explained
from src.cf_transport import get_dice_transformed, get_closest_target
from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
# We will be using data from  the UCI: Breast Cancer Wisconsin (Original)

all_toxic_labels = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']

def save_objs(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)
def load_objs(file):
    with open(file, "rb") as f:
        objs = pickle.load(f)
    return objs

def get_gender_predictions(tokenizer, model, text):
    tokenizer_out = tokenizer(text)
    
    input_ids = tokenizer_out["input_ids"]
    mask = tokenizer_out["attention_mask"]
    model_out = model(torch.tensor(input_ids).view(1,-1), torch.tensor(mask).view(1,-1))
    pred_scores = F.softmax(model_out.logits, dim=-1) 
    return pred_scores.view(-1)[1]


def select_samples_by_clustering(X0, group_idx_ls1, count = 2000):

    
    all_selected_sample_id_ls = []

    for group_idx in group_idx_ls1:
        cluster_count = int(count/len(group_idx_ls1))
        kmeans_func = KMeans(n_clusters=cluster_count, random_state=0)
        sub_data = X0[group_idx]
        kmeans = kmeans_func.fit(sub_data)
        cluster_centers = kmeans.cluster_centers_

        full_dist = np.sum((sub_data.reshape(sub_data.shape[0], 1, -1) - cluster_centers.reshape(1, cluster_centers.shape[0], -1))**2, axis = -1)

        cluster_closest_sample_ids = np.argmin(full_dist, axis = 0)

        all_selected_sample_id_ls.append(cluster_closest_sample_ids)

    all_selected_sample_id_array = np.concatenate(all_selected_sample_id_ls, axis=-1)

    return all_selected_sample_id_array
        

        





        

def train_classifying_source_target_model_regular(source,target, source_groups, target_groups):
    
    # net = PTNN(source.shape[1], 1)
    net = PTLogisticRegression(source.shape[1], 1)
    
    x_train = torch.from_numpy(np.concatenate([source, target], axis=0)).float()
    x_train = x_train / torch.max(x_train, dim=0)[0]
    x_train[torch.isnan(x_train)] = 0
    group_mat = np.concatenate([source_groups, target_groups], axis=0)
    no_group = 1 - (np.sum(group_mat, axis=1) > 0)[:, np.newaxis]
    print(no_group)
    all_group = np.ones_like(no_group)
    group_mat = torch.from_numpy(np.concatenate([group_mat, no_group], axis=1)).double()
    # group_mat = torch.from_numpy(group_mat).double()

    y = torch.from_numpy(np.concatenate([np.ones(source.shape[0]), np.zeros(target.shape[0])], axis=0)).float()[:, None]
    print("Input shape:", x_train.shape)
    regular_training(
        net,
        x_train,
        y,
        group_mat,
        200,
        0.5,
        5e-5)
    
    return net
    

def select_target_samples(target_sample_count, net1, net2, all_target_samples):
    all_target_prob1 = net1(torch.from_numpy(all_target_samples).type(torch.float))
    all_target_prob2 = net2(torch.from_numpy(all_target_samples).type(torch.float))
    if len(all_target_prob1.shape) >= 2 and all_target_prob1.shape[1] > 1:
        all_target_prob1 = all_target_prob1[:,0]
        all_target_prob2 = all_target_prob2[:,0]
    average_target_prob = (all_target_prob1 + all_target_prob2)/2
    average_dist = torch.abs(average_target_prob.view(-1) - 0.5)
    sorted_sample_ids = torch.argsort((average_dist), descending=True).numpy()
    selected_sample_ids = sorted_sample_ids[0:target_sample_count]
    return all_target_samples[selected_sample_ids], selected_sample_ids
    

def train_classifying_source_target_model_group(source, target, source_groups,target_groups):
    
    # net = PTNN_nlp(source.shape[1], 1)
    net = PTLogisticRegression(source.shape[1], 1)
    group_mat = np.concatenate([source_groups, target_groups], axis=0)
    print(np.sum(group_mat > 1))
    no_group = 1 - (np.sum(group_mat, axis=1) > 0)[:, np.newaxis]
    print(no_group)
    all_group = np.ones_like(no_group)
    group_mat = torch.from_numpy(np.concatenate([group_mat, no_group], axis=1)).double()
    # group_mat = torch.from_numpy(group_mat).double()
    X_train = torch.from_numpy(np.concatenate([source, target], axis=0)).float()
    X_train = X_train / torch.max(X_train, dim=0)[0]
    X_train[torch.isnan(X_train)] = 0
    y = torch.from_numpy(np.concatenate([np.ones(source.shape[0]), np.zeros(target.shape[0])], axis=0)).float()[:, None]
    print(group_mat)
    dro_training(net, 
        X_train,
        y,
        group_mat,
        200,
        0.5,
        5e-5)
    
    return net



def downsample_data(selected_idx, demographics_data, demographics_groupds, demographics_label, other_data, other_groups, other_labels, demo1, full_data, full_groups, full_labels):
    
    
    # if len(demographics_data[demo1]) > count:
        

    all_sample_idx = np.ones(len(demographics_data[demo1]))
    all_sample_idx[selected_idx] = 0
    other_sample_idx = np.nonzero(all_sample_idx)[0].reshape(-1)

    full_data[demo1] = np.copy(demographics_data[demo1])
    full_groups[demo1] = torch.clone(demographics_groupds[demo1])
    full_labels[demo1] = np.copy(demographics_label[demo1])
    
    other_data[demo1] = demographics_data[demo1][other_sample_idx]
    other_groups[demo1] = demographics_groupds[demo1][other_sample_idx]
    other_labels[demo1] = demographics_label[demo1][other_sample_idx]
    demographics_data[demo1] = demographics_data[demo1][selected_idx]
    demographics_groupds[demo1] = demographics_groupds[demo1][selected_idx]
    demographics_label[demo1] = demographics_label[demo1][selected_idx]

def get_all_words_from_all_sentences(sentence_ls):
    all_word_set = set()
    for sentence in sentence_ls:
        simplified_sentence = re.sub("[^a-zA-Z ]+", "", sentence)
        word_ls = simplified_sentence.split(" ")
        all_word_set.update(word_ls)

    return list(all_word_set)

def get_all_words_related_to_a_topic(all_word_embedding_tensor, topic_word, word_count=500):
    topic_word_embedding = torch.tensor(get_word_embedding(topic_word))
    topic_word_embedding = topic_word_embedding.view(1,-1)
    sim_scores = torch.sum(all_word_embedding_tensor.unsqueeze(1)*topic_word_embedding.unsqueeze(0), dim=-1)
    sim_scores = sim_scores.view(-1)

    sorted_scores, sorted_indices = torch.sort(sim_scores, descending=True)

    selected_indices = sorted_indices[0:word_count]

    return selected_indices


def construct_vocab_for_counter(word_ls, selected_word_indices_ls):
    vocab = {}
    for idx in range(len(selected_word_indices_ls)):
        selected_word = word_ls[selected_word_indices_ls[idx]]
        vocab[selected_word] = idx
    return vocab

def get_demographic_counts(is_subdemographic, save_dir, demo1, demo2, demographics_data,demographics_groups,
                           ngram, max_features, equalize_sizes=False, sort_word=False):
    # splitting between demographic attribute 1 and 2
    og_X0 = demographics_data[demo1]
    og_X0_group = demographics_groups[demo1]
    og_X1 = demographics_data[demo2]
    og_X1_group = demographics_groups[demo2]
    if equalize_sizes:
        print('Equalizing sizes.')
        print('OG sizes:', (og_X0.shape, og_X1.shape))
        if og_X0.shape[0] < og_X1.shape[0]:
            rng = np.random.RandomState(42)
            # if not os.path.exists(os.path.join(save_dir, "subsample_idxs")):
            subsample_idxs = rng.choice(len(og_X1), replace=False, size=len(og_X0))
            # else:
            #     subsample_idxs  = load_objs(os.path.join(save_dir, "subsample_idxs"))            
            og_X1 = og_X1[subsample_idxs]  # subsampling to equalize sizes
            og_X1_group = og_X1_group[subsample_idxs]
        elif og_X0.shape[0] > og_X1.shape[0]:
            rng = np.random.RandomState(42)
            # if not os.path.exists(os.path.join(save_dir, "subsample_idxs")):
            subsample_idxs = rng.choice(len(og_X0), replace=False, size=len(og_X1))
            # else:
            #     subsample_idxs  = load_objs(os.path.join(save_dir, "subsample_idxs"))  
            # subsample_idxs = rng.choice(len(og_X0), replace=False, size=len(og_X1))
            og_X0 = og_X0[subsample_idxs]
            og_X0_group = og_X0_group[subsample_idxs]
        print('New sizes:', (og_X0.shape, og_X1.shape))
    og_X_both = np.concatenate((og_X0, og_X1))

    if sort_word:
        # if not os.path.exists(os.path.join(save_dir, "word_ls")):
        word_ls = get_all_words_from_all_sentences(og_X_both.tolist())
        #     save_objs(word_ls, os.path.join(save_dir, "word_ls"))
        # else:
        #     word_ls = load_objs(os.path.join(save_dir, "word_ls"))

        if not os.path.exists(os.path.join(save_dir, "toxic_probs")):
            # all_word_embeddings = get_word_embeddings_for_all(word_ls)
            all_word_pred_ls = get_toxic_predictions_for_word_ls(word_ls)
            toxic_probs = all_word_pred_ls[:,0]
            save_objs(toxic_probs, os.path.join(save_dir, "toxic_probs"))
        else:
            toxic_probs = load_objs(os.path.join(save_dir, "toxic_probs"))

        sorted_ids = torch.argsort(toxic_probs, descending=True)
        selected_word_indices_ls = sorted_ids[0:max_features]
        
        
        vocab = construct_vocab_for_counter(word_ls, selected_word_indices_ls)

        ngram_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),
                                       stop_words='english', max_features=max_features, vocabulary=vocab)
        # if is_subdemographic:
        #     selected_word_indices = get_all_words_related_to_a_topic(all_word_embeddings, "toxic", max_features)
        # else:
        #     selected_word_indices = get_all_words_related_to_a_topic(all_word_embeddings, "female", max_features)

    # selected_word_indices_ls = selected_word_indices.tolist()
    else:
        
        ngram_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),
                                        stop_words='english') #, max_features=max_features)
    og_X_both = np.concatenate((og_X0, og_X1))
    
    vectorized_data = ngram_vectorizer.fit_transform(og_X_both).toarray()

    

    # transformed_vectorized_data = maxabs_transformer.transform(vectorized_data.toarray().astype(float))
    # X0, X1 = transformed_vectorized_data[:og_X0.shape[0]], transformed_vectorized_data[og_X0.shape[0]:] 
    X0, X1 = vectorized_data[:og_X0.shape[0]].astype(float), vectorized_data[og_X0.shape[0]:].astype(float)
    feature_name_ls = ngram_vectorizer.get_feature_names_out()

    X_combined = np.concatenate([X0, X1], axis=0)
    sel = SelectKBest(chi2, k=50).fit(X_combined, np.concatenate([np.zeros(X0.shape[0]), np.ones(X1.shape[0])], axis=0))
    feature_name_ls = list(feature_name_ls[sel.get_support()])
    X0 = X0[:, sel.get_support()]
    X1 = X1[:, sel.get_support()]
    # nonempty_feat_ids = (np.sum(np.concatenate([X0, X1], axis=0), axis=0) > 0)
    # feature_name_ls = [feature_name_ls[idx] for idx in nonempty_feat_ids]
    # X0 = X0[:, nonempty_feat_ids]
    # X1 = X1[:, nonempty_feat_ids]
    
    return X0, X1, feature_name_ls, og_X0, og_X1, og_X0_group, og_X1_group


def transform_sentences_to_vectors(orig_X0, ngram_vectorizer):
    X0 = ngram_vectorizer.transform(orig_X0)
    return X0.toarray().astype(float)


def compute_explanations_for_dice(seed, X0, X1, selected_X0_ids, full_X0, full_X1, X0_group, X1_group, full_X1_group, save_dir, feature_name_ls, net_for_dice, group_method=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    full_data = np.concatenate([full_X0, full_X1], axis = 0)
    # feature_name_ls = list(ngram_vectorizer.get_feature_names_out())
    full_df_dict = {}
    for idx in range(len(feature_name_ls)):
        feature_col = full_data[:,idx]
        feature_name = feature_name_ls[idx]
        full_df_dict[feature_name] = feature_col
    
    labels = np.concatenate([np.ones(len(full_X0)), np.zeros(len(full_X1))])
    full_df_dict["_label"] = labels
    full_df = pd.DataFrame(full_df_dict, dtype=float)
    source_df = full_df.iloc[0:len(full_X0)]
    target_df = full_df.iloc[len(full_X0):]
    
    x_s = get_dice_transformed(
            net_for_dice,
            full_df / full_df.max(),
            (source_df / full_df.max()).drop(columns=['_label']).iloc[selected_X0_ids],
            "_label",
            0)
    
    norm_const = full_df.drop(columns=["_label"]).max().to_numpy().astype(float)
    print("MIN:", full_df.drop(columns=["_label"]).min().to_numpy().astype(float))
    x_s_norm = (x_s.to_numpy().astype(float) * norm_const)
    # if mapping is None:
    #     mapping = get_closest_target(x_s_norm, full_X1)

    group_percent_explained(
        X0[selected_X0_ids],
        x_s_norm,
        X1,
        X0_group.numpy()[selected_X0_ids],
        X1_group.numpy(),
        ["c1", "c2", "c3", "c4", "c5", "c6"])
    
    if group_method:
        with open(os.path.join(save_dir, f"group_x_s_{seed}"), "wb") as f:
            pickle.dump(x_s_norm, f)

    else:
        with open(os.path.join(save_dir, f"origin_x_s_{seed}"), "wb") as f:
            pickle.dump(x_s_norm, f)
            

def compute_explanations(index, args, X0, X1, X0_group, X1_group, save_dir, feature_name_ls, n_features, kmeans_cluster_count=4, net_for_dice=None):
    no_group_s = np.ones((X0.shape[0], 1))
    no_group_t = np.ones((X1.shape[0], 1))

    if args.method == "ot":
        x_s = group_feature_transport(X0, X1, no_group_s, no_group_t, n_features, 0.1, 200)
        
    elif args.method == "kmeans":
        # vectorized_data = np.concatenate([X0, X1], axis = 0)
        # maxabs_transformer = MaxAbsScaler().fit(vectorized_data.astype(float))
        # transformed_vectorized_data = maxabs_transformer.transform(vectorized_data.astype(float))
        # X0 = transformed_vectorized_data[0:X0.shape[0]]
        # X1 = transformed_vectorized_data[X0.shape[0]:]
        x_s, centroids, shifts = group_kmeans_shift_transport(X0, X1, no_group_s, no_group_t,
                n_features, clusters=kmeans_cluster_count, lr=20, iters=200)
        shifts = shifts.round(0)
        x_s = transform_samples_kmeans(X0, centroids, shifts)
        # x_s = maxabs_transformer.inverse_transform(x_s)
        # X0 = maxabs_transformer.inverse_transform(X0)
        # X1 = maxabs_transformer.inverse_transform(X1)
        with open(os.path.join(save_dir, f"centroids_{index}"), "wb") as f:
            pickle.dump(centroids, f)
        with open(os.path.join(save_dir, f"shifts_{index}"), "wb") as f:
            pickle.dump(shifts, f)
        
    elif args.method == 'mean':
        x_s,_ = group_mean_shift_transport(X0, X1, no_group_s, no_group_t, n_features, 1, 500)
    
    x_s = x_s.round(0)
    group_percent_explained(X0, x_s, X1, X0_group.numpy(), X1_group.numpy(), feature_name_ls)
    with open(os.path.join(save_dir, f"origin_x_s_{index}"), "wb") as f:
        pickle.dump(x_s, f)

    if args.method == "ot":
        x_s = group_feature_transport(X0, X1, X0_group.numpy(), X1_group.numpy(), n_features, 0.1, 200)
    elif args.method == "kmeans":
        x_s, centroids, shifts = group_kmeans_shift_transport(X0, X1, X0_group.numpy(),
                X1_group.numpy(), n_features, clusters=kmeans_cluster_count,
                lr=20, iters=200)
        shifts = shifts.round(0)
        with open(os.path.join(save_dir, f"group_centroids_{index}"), "wb") as f:
            pickle.dump(centroids, f)
        with open(os.path.join(save_dir, f"group_shifts_{index}"), "wb") as f:
            pickle.dump(shifts, f)
    elif args.method == 'mean':
        x_s,_ = group_mean_shift_transport(X0, X1, X0_group.numpy(), X1_group.numpy(), n_features, lr=1, iters =500)
            
    
    x_s = x_s.round(0)


    group_percent_explained(X0, x_s, X1, X0_group.numpy(), X1_group.numpy(), feature_name_ls)


    with open(os.path.join(save_dir, f"group_x_s_{index}"), "wb") as f:
        pickle.dump(x_s, f)
    with open(os.path.join(save_dir, f"feature_name_ls_{index}"), "wb") as f:
        pickle.dump(feature_name_ls, f)



def compute_transformed_diff(x_s, X0, origin_X0, ngram_vectorizer):
    
    feat_name_ls = list(ngram_vectorizer.get_feature_names_out())
    diff = x_s - X0

    origin_word_id_ls = np.nonzero(X0[0] != 0)[0]
    origin_word_count_ls = X0[0][origin_word_id_ls]
    origin_word_ls = [feat_name_ls[idx] for idx in origin_word_id_ls]
    x_s = (x_s + 0.5).astype(np.int32)
    new_word_id_ls = np.nonzero(x_s[0] != 0)[0]
    new_word_count_ls = x_s[0][new_word_id_ls]
    new_word_ls = [feat_name_ls[idx] for idx in new_word_id_ls]

    print(origin_X0[0])
    print()
    print(origin_word_ls)
    print(origin_word_count_ls)
    print()
    print(new_word_ls)
    print(new_word_count_ls)
    print()





def retrieve_api_key_file(api_key_file):
    with open(api_key_file) as f:
        json_key = json.load(f)    

    return json_key["key"]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='pre_process_and_train_data.py [<args>] [-h | --help]'
    )
    
    # parser.add_argument('--epochs', type=int, default=200, help='used for resume')
    # # parser.add_argument('--batch_size', type=int, default=4096, help='used for resume')
    # # parser.add_argument('--lr', type=float, default=0.02, help='used for resume')

    # parser.add_argument('--batch_size', type=int, default=64, help='used for resume')
    # parser.add_argument('--lr', type=float, default=0.002, help='used for resume')
    # parser.add_argument('--model', type=str, default=0.02, help='used for resume', choices=["mlp", "dd"])
    parser.add_argument('--train', action='store_true', help='use GPU')
    parser.add_argument('--method', type=str,choices=["ot", "kmeans", "mean", "dice"], help='use GPU', default="ot")
    parser.add_argument('--api_key_file', type=str, default=None, help='used for resume')
    parser.add_argument('--kmeans_cluster_count', type=int, default=4, help='used for resume')
    parser.add_argument('--data_dir', type=str, default="out/", help='used for resume')
    parser.add_argument('--no_groups', action='store_true')

    args = parser.parse_args()
    return args

# def get_word_embedding(word):
#     response = openai.Embedding.create(
#         input=word,
#         model="text-embedding-ada-002"
#     )
#     embeddings = response['data'][0]['embedding']
#     return embeddings


def get_toxic_predictions(word):
    results = Detoxify('original').predict(word)
    return results

def get_toxic_predictions_for_all(word_ls):
    results_ls = []
    for word in tqdm(word_ls):
        results = get_toxic_predictions(word)
        
        results_ls.append(convert_predictions_to_array(results).view(-1))
    return torch.stack(results_ls,dim=0)


def get_toxic_predictions_for_word_ls(word_ls):
    results_ls = []
    # for word in tqdm(word_ls):
    bz = 200
    for start_id in tqdm(range(0, len(word_ls), 200)):
        end_id = start_id + bz
        if end_id >= len(word_ls):
            end_id = len(word_ls)
        sub_word_ls = word_ls[start_id:end_id]
        results = Detoxify('original').predict(sub_word_ls)
        
        results_ls.append(convert_predictions_to_array(results).t())
    return torch.cat(results_ls,dim=0)

# def get_word_embeddings_for_all(word_ls):
#     embedding_ls = []
#     for word in tqdm(word_ls):
#         embedding = get_word_embedding(word)
#         embedding_ls.append(torch.tensor(embedding).view(-1))
#     return torch.stack(embedding_ls, dim=0)


# def get_word_embeddings_for_all_negated(word_ls):
#     embedding_ls = []
#     for word in word_ls:
#         embedding = get_word_embedding("not " + word)
#         embedding_ls.append(torch.tensor(embedding).view(-1))
#     return torch.stack(embedding_ls, dim=0)

# def query_llm(text):
#     while(True):
#         success=False
#         try:
#             response = openai.Completion.create(
#                 model="text-davinci-003",
#                 prompt=text,
#                 temperature=0,
#                 max_tokens=1024,
#                 top_p=1,
#                 frequency_penalty=0,
#                 presence_penalty=0
#             )
#             success = True
#         except:
#             Time.sleep(30)
#             continue
            
        
#         if success:
#             break

#     answer = response["choices"][0]["text"]
#     answer = answer.strip()

#     return answer


# def get_response_from_llm(text):
#     answer = query_llm(text)

#     if "true" in answer.strip().lower() or "yes" in answer.strip().lower():
#         return answer, True
#     else:
#         return answer, False

# def check_toxic_with_gpt(orig_X_text):
#     orig_X_text = orig_X_text.strip().replace("\"", "")
#     q_text = "\"" + orig_X_text + ".\"\n" + "Is the above quoted paragraph toxic? True or False."

#     # q_text = "Is the following sentence toxic? True or False. \"" + orig_X_text + "\""
#     answer, gpt_response = get_response_from_llm(q_text)
#     return gpt_response


# def check_female_with_gpt(orig_X_text):
#     orig_X_text = orig_X_text.strip().replace("\"", "")
#     q_text = "\"" + orig_X_text + ".\"\n" + "Is the above quoted paragraph about female? True or False."

#     # q_text = "Is the following sentence toxic? True or False. \"" + orig_X_text + "\""
#     answer, gpt_response = get_response_from_llm(q_text)
#     return gpt_response


# def add_word_with_gpt(text, word):
#     text = text.strip().replace('\n','').replace('"','')
#     query_text = "\"" + text + "\"\n" + " Insert one word \"" + word + "\" to the above quoted paragraph and output the whole paragraph:\n"
#     # print("query text::")
#     # print(query_text)
#     answer = query_llm(query_text)
#     return answer


def obtain_remaining_words_from_sentence(text, word_ls):

    remaining_word_ls = []
    for word in word_ls:
        if word in text:
            text = text.replace(word, "", 1)
        else:
            remaining_word_ls.append(word)
    return remaining_word_ls
        

# def add_word_ls_with_gpt(text, word, word_count):
#     text = text.strip().replace('\n','').replace('"','')

#     query_text = "\"" + text + "\"\n" + " Insert the following " + str(word_count) + " quoted word " + word + " to the above quoted paragraph and output the whole paragraph:\n"
#     # print("query text::")
#     # print(query_text)
#     answer = query_llm(query_text)
#     return answer


# def remove_word_with_gpt(text, word):
#     text = text.strip().replace('"', "")
#     query_text = "\"" + text + "\"\n" + " Delete one word \"" + word + "\" from the above quoted paragraph and output the whole paragraph:" + "\n" +'"""'
#     # print("query text::")
#     # print(query_text)
#     answer = query_llm(query_text)
#     return answer

def change_single_sentence(origin_X0, word_ls, diff):
    add_word_idx = np.nonzero(diff > 0)[0]
    added_word_ls = [word_ls[idx] for idx in add_word_idx]
    added_word_count_ls = diff[add_word_idx]

    violation = False

    remove_word_idx = np.nonzero(diff < 0)[0]

    remove_word_ls = [word_ls[idx] for idx in remove_word_idx]

    word_len_ls = torch.tensor([len(word) for word in remove_word_ls])

    sorted_idx = torch.argsort(word_len_ls, descending=True)

    sorted_idx = sorted_idx.view(-1).tolist()

    sorted_remove_word_idx = [remove_word_idx[idx] for idx in sorted_idx]
    sorted_remove_word_ls = [word_ls[idx] for idx in sorted_remove_word_idx]

    remove_word_count_ls = diff[sorted_remove_word_idx]
    for word_idx in range(len(sorted_remove_word_ls)):
        word = sorted_remove_word_ls[word_idx]
        remove_word_count = remove_word_count_ls[word_idx]
        for k in range(np.abs(int(remove_word_count))):
            if word in origin_X0:
                origin_X0 = origin_X0.replace(" " + word, "", 1)
            else:
                violation = True

    # assert violation == False

    add_word_idx = np.nonzero(diff > 0)[0]
    added_word_ls = [word_ls[idx] for idx in add_word_idx]
    added_word_count_ls = diff[add_word_idx]
    prefix = ""
    
    for word_idx in range(len(added_word_ls)):
        add_word = added_word_ls[word_idx]
        add_word_count = int(added_word_count_ls[word_idx])
        for k in range(add_word_count):
            if k > 0:
                prefix += " "
            prefix += add_word
    
    origin_X0 = prefix + " " + origin_X0
    return origin_X0, violation


def constructing_sentence_with_words(x_s, word_ls):
    word_idx = np.nonzero(x_s > 0)[0]
    added_word_ls = [word_ls[idx] for idx in word_idx]
    origin_X0 = ""
    added_word_count_ls = x_s[word_idx]
    for word_idx in range(len(added_word_ls)):
        add_word = added_word_ls[word_idx]
        add_word_count = int(added_word_count_ls[word_idx])
        for k in range(add_word_count):
            origin_X0 += " " + add_word
    return origin_X0, 0

def change_single_sentence2(origin_X0, word_ls, diff):
    add_word_idx = np.nonzero(diff > 0)[0]
    added_word_ls = [word_ls[idx] for idx in add_word_idx]
    added_word_count_ls = diff[add_word_idx]

    violation = False


    remove_word_idx = np.nonzero(diff < 0)[0]
    remove_word_ls = [word_ls[idx] for idx in remove_word_idx]
    remove_word_count_ls = diff[remove_word_idx]
    for word_idx in range(len(remove_word_ls)):
        word = remove_word_ls[word_idx]
        remove_word_count = remove_word_count_ls[word_idx]
        for k in range(np.abs(int(remove_word_count))):
            # origin_X0 = remove_word_with_gpt(origin_X0, word)
            if word in origin_X0:
                origin_X0 = origin_X0.replace(word, "", 1)


    add_word_idx = np.nonzero(diff > 0)[0]
    added_word_ls = [word_ls[idx] for idx in add_word_idx]
    added_word_count_ls = diff[add_word_idx]
    print("original sentence::")
    print(origin_X0)

    # for word_idx in range(len(added_word_ls)):
    #     add_word = added_word_ls[word_idx]
    #     add_word_count = int(added_word_count_ls[word_idx])
    #     for k in range(add_word_count):
    #         # origin_X0 += " " + add_word
    #         # Time.sleep(5)
    #         origin_X0 = add_word_with_gpt(origin_X0, add_word)

    all_add_word_ls = ""
    all_add_word_count = 0
    all_word_ls = []
    for word_idx in range(len(added_word_ls)):
        add_word = added_word_ls[word_idx]
        add_word_count = int(added_word_count_ls[word_idx])
        for k in range(add_word_count):
            if all_add_word_count >= 1:
                all_add_word_ls += ","

            all_add_word_ls += '"' + add_word + '"'
            all_add_word_count += 1
            all_word_ls.append(add_word)

    
    origin_X0 = add_word_ls_with_gpt(origin_X0, all_add_word_ls, all_add_word_count)
    remaining_word_ls = obtain_remaining_words_from_sentence(origin_X0.lower(), all_word_ls)
    prev_remaining_word_count = len(remaining_word_ls)

    wait_time = 0

    while len(remaining_word_ls) > 0:
        all_add_word_ls = ""
        all_add_word_count = 0
        for k in range(len(remaining_word_ls)):
            add_word = remaining_word_ls[k]
            if all_add_word_count >= 1:
                all_add_word_ls += ","

            all_add_word_ls += '"' + add_word + '"'
            all_add_word_count += 1


        origin_X0 = add_word_ls_with_gpt(origin_X0, all_add_word_ls, all_add_word_count)
        remaining_word_ls = obtain_remaining_words_from_sentence(origin_X0.lower(), all_word_ls)

        curr_remaining_word_count = len(remaining_word_ls)

        if curr_remaining_word_count >= prev_remaining_word_count:
            wait_time += 1
            if wait_time >= 3:
                for remaining_word in remaining_word_ls:
                    origin_X0 += " " + remaining_word

                print("cannot add %d words in this sentence"%(curr_remaining_word_count))
                break
        else:
            wait_time = 0
        prev_remaining_word_count = curr_remaining_word_count

    print("modified sentence::")
    print(origin_X0)
    
    
    
    
    return origin_X0, violation

def convert_predictions_to_array(prediction_map):
    prediction_score_ls = []
    for key in all_toxic_labels:
        prediction_score = prediction_map[key]
        prediction_score_ls.append(prediction_score)
    prediction_score_tensor = torch.tensor(prediction_score_ls)
    return prediction_score_tensor

def edit_text_based_on_explanation(X0, X0_prediction_scores, X_s, group_X_s, origin_X0, word_ls, x0_group, gender_tokenizer, gender_model):
    X_s = (X_s + 0.5).astype(np.int32)
    group_X_s = (group_X_s + 0.5).astype(np.int32)
    diff = X_s - X0
    group_diff = group_X_s - X0
    
    modified_X0_embedding_ls=[]
    group_modified_X0_embedding_ls=[]
    
    related_score_changed = 0
    related_score_changed_group = 0



    for idx in range(len(origin_X0)):
        X_text = origin_X0[idx]
        # is_origin_toxic = check_toxic_with_gpt(X_text)

        curr_diff = diff[idx]

        curr_group_diff = group_diff[idx]

        print("editing sample::", idx)
        X_text = X_text.lower()
        modified_X_text,_ = change_single_sentence(X_text, word_ls, curr_diff)
        group_modified_X_text,_ = change_single_sentence(X_text, word_ls, curr_group_diff)

        # X_text_embedding = torch.tensor(get_word_embedding(X_text))
        X_text_prediction_modified = get_toxic_predictions(modified_X_text)
        X_text_prediction_group_modified = get_toxic_predictions(group_modified_X_text)


        # female_pred_score = get_gender_predictions(gender_tokenizer, gender_model, modified_X_text)
        # group_female_pred_score = get_gender_predictions(gender_tokenizer, gender_model, group_modified_X_text)


        modified_X_text_prediction_scores = convert_predictions_to_array(X_text_prediction_modified)
        group_modified_X_text_prediction_scores = convert_predictions_to_array(X_text_prediction_group_modified)

        # X_text_embedding_modified = torch.tensor(get_word_embedding(modified_X_text))
        # X_text_embedding_group_modified = torch.tensor(get_word_embedding(group_modified_X_text))

        modified_X0_embedding_ls.append(modified_X_text_prediction_scores.view(-1))
        group_modified_X0_embedding_ls.append(group_modified_X_text_prediction_scores.view(-1))

    #     if x0_group[idx] == 1:
    #         # origin_sim = torch.sum(X0_prediction_scores[idx]*selected_topic_embeddings)
    #         # modified_sim = torch.sum(X_text_embedding_modified*selected_topic_embeddings)
    #         # group_modified_sim = torch.sum(X_text_embedding_group_modified*selected_topic_embeddings)
    #         related_score_changed += (female_pred_score - 1)**2
    #         related_score_changed_group += (group_female_pred_score - 1)**2

    # print("modified related score::", related_score_changed)
    # print("group modified related score::", related_score_changed_group)
    # print()

    modified_X0_embedding_tensor = torch.stack(modified_X0_embedding_ls)
    group_modified_X0_embedding_tensor = torch.stack(group_modified_X0_embedding_ls)
    return modified_X0_embedding_tensor, group_modified_X0_embedding_tensor

# def edit_text_based_on_explanation2(X0, X0_embeddings, X_s, group_X_s, origin_X0, word_ls, selected_topic_embeddings, x0_group):
#     X_s = (X_s + 0.5).astype(np.int32)
#     group_X_s = (group_X_s + 0.5).astype(np.int32)#     # diff = X_s - X0
#     # group_diff = group_X_s - X0
    
#     modified_X0_embedding_ls=[]
#     group_modified_X0_embedding_ls=[]
    
#     related_score_changed = 0
#     related_score_changed_group = 0

#     for idx in range(len(origin_X0)):
#         X_text = origin_X0[idx]
#         # is_origin_toxic = check_toxic_with_gpt(X_text)

#         curr_x_s = X_s[idx]
#         curr_group_x_s = group_X_s[idx]

#         print("editing sample::", idx)
#         X_text = X_text.lower()

#         modified_X_text,_  =constructing_sentence_with_words(curr_x_s, word_ls)
#         # modified_X_text,_ = change_single_sentence(X_text, word_ls, curr_diff)
#         group_modified_X_text,_ = constructing_sentence_with_words(curr_group_x_s, word_ls)
#         # group_modified_X_text,_ = change_single_sentence(X_text, word_ls, curr_group_diff)

#         # X_text_embedding = torch.tensor(get_word_embedding(X_text))
#         X_text_embedding_modified = torch.tensor(get_word_embedding(modified_X_text))
#         X_text_embedding_group_modified = torch.tensor(get_word_embedding(group_modified_X_text))

#         modified_X0_embedding_ls.append(X_text_embedding_modified.view(-1))
#         group_modified_X0_embedding_ls.append(X_text_embedding_group_modified.view(-1))

#         if x0_group[idx] == 1:
#             origin_sim = torch.sum(X0_embeddings[idx]*selected_topic_embeddings)
#             modified_sim = torch.sum(X_text_embedding_modified*selected_topic_embeddings)
#             group_modified_sim = torch.sum(X_text_embedding_group_modified*selected_topic_embeddings)
#             related_score_changed += modified_sim - origin_sim
#             related_score_changed_group += group_modified_sim - origin_sim

#     print("modified related score::", related_score_changed)
#     print("group modified related score::", related_score_changed_group)
#     print()

#     modified_X0_embedding_tensor = torch.stack(modified_X0_embedding_ls)
#     group_modified_X0_embedding_tensor = torch.stack(group_modified_X0_embedding_ls)
#     return modified_X0_embedding_tensor, group_modified_X0_embedding_tensor


def edit_text_based_on_explanation_and_predict_labels(X0, X_s, group_X_s, origin_X0, word_ls, is_toxic):
    X_s = (X_s + 0.5).astype(np.int32)
    group_X_s = (group_X_s + 0.5).astype(np.int32)
    diff = X_s - X0
    group_diff = group_X_s - X0
    
    changed_idx_ls = []
    group_changed_idx_ls=[]

    is_origin_toxic=False
    changed_pred_count = 0
    group_changed_pred_count = 0
    for idx in range(len(origin_X0)):
        X_text = origin_X0[idx]
        # is_origin_toxic = check_toxic_with_gpt(X_text)

        curr_diff = diff[idx]

        curr_group_diff = group_diff[idx]

        print("editing sample::", idx)
        X_text = X_text.lower()
        modified_X_text,_ = change_single_sentence(X_text, word_ls, curr_diff)
        group_modified_X_text,_ = change_single_sentence(X_text, word_ls, curr_group_diff)

        if is_toxic:

            is_modified_toxic = check_toxic_with_gpt(modified_X_text)

            is_group_modified_toxic = check_toxic_with_gpt(group_modified_X_text)
        else:
            is_modified_toxic = check_female_with_gpt(modified_X_text)

            is_group_modified_toxic = check_female_with_gpt(group_modified_X_text)


        changed_pred_count += (is_modified_toxic != is_origin_toxic)

        group_changed_pred_count += (is_group_modified_toxic != is_origin_toxic)

        if is_group_modified_toxic != is_origin_toxic:
            group_changed_idx_ls.append(idx)
        if is_modified_toxic != is_origin_toxic:
            changed_idx_ls.append(idx)
        print("curr changed samples::", changed_pred_count)

        print("curr changed id ls::", changed_idx_ls)

        print("curr group changed samples::", group_changed_pred_count)

        print("curr group changed id ls::", group_changed_idx_ls)

    print("changed samples::", changed_pred_count)

    print("group changed samples::", group_changed_pred_count)

    return changed_pred_count, group_changed_pred_count

def edit_text_based_on_explanation_and_predict_labels2(X0, X_s, group_X_s, origin_X0, word_ls, is_toxic):
    X_s = (X_s + 0.5).astype(np.int32)
    group_X_s = (group_X_s + 0.5).astype(np.int32)
    # diff = X_s - X0
    # group_diff = group_X_s - X0
    
    changed_idx_ls = []
    group_changed_idx_ls=[]

    is_origin_toxic=False
    changed_pred_count = 0
    group_changed_pred_count = 0
    for idx in range(len(origin_X0)):
        X_text = origin_X0[idx]
        # is_origin_toxic = check_toxic_with_gpt(X_text)

        curr_diff = X_s[idx]

        curr_group_diff = group_X_s[idx]

        print("editing sample::", idx)
        X_text = X_text.lower()
        modified_X_text,_ = constructing_sentence_with_words(curr_diff, word_ls)
        group_modified_X_text,_ = constructing_sentence_with_words(curr_group_diff, word_ls)

        if is_toxic:

            is_modified_toxic = check_toxic_with_gpt(modified_X_text)

            is_group_modified_toxic = check_toxic_with_gpt(group_modified_X_text)
        else:
            is_modified_toxic = check_female_with_gpt(modified_X_text)

            is_group_modified_toxic = check_female_with_gpt(group_modified_X_text)


        changed_pred_count += (is_modified_toxic != is_origin_toxic)

        group_changed_pred_count += (is_group_modified_toxic != is_origin_toxic)

        if is_group_modified_toxic != is_origin_toxic:
            group_changed_idx_ls.append(idx)
        if is_modified_toxic != is_origin_toxic:
            changed_idx_ls.append(idx)
        print("curr changed samples::", changed_pred_count)

        print("curr changed id ls::", changed_idx_ls)

        print("curr group changed samples::", group_changed_pred_count)

        print("curr group changed id ls::", group_changed_idx_ls)

    print("changed samples::", changed_pred_count)

    print("group changed samples::", group_changed_pred_count)

    return changed_pred_count, group_changed_pred_count
        # X_text_embedding = torch.tensor(get_word_embedding(X_text))
        # X_text_embedding_modified = torch.tensor(get_word_embedding(modified_X_text))
        # X_text_embedding_group_modified = torch.tensor(get_word_embedding(group_modified_X_text))

        # modified_X0_embedding_ls.append(X_text_embedding_modified.view(-1))
        # group_modified_X0_embedding_ls.append(X_text_embedding_group_modified.view(-1))

    #     if x0_group[idx] == 1:
    #         origin_sim = torch.sum(X0_embeddings[idx]*selected_topic_embeddings)
    #         modified_sim = torch.sum(X_text_embedding_modified*selected_topic_embeddings)
    #         group_modified_sim = torch.sum(X_text_embedding_group_modified*selected_topic_embeddings)
    #         related_score_changed += modified_sim - origin_sim
    #         related_score_changed_group += group_modified_sim - origin_sim

    # print("modified related score::", related_score_changed)
    # print("group modified related score::", related_score_changed_group)
    # print()

    # modified_X0_embedding_tensor = torch.stack(modified_X0_embedding_ls)
    # group_modified_X0_embedding_tensor = torch.stack(group_modified_X0_embedding_ls)
    # return modified_X0_embedding_tensor, group_modified_X0_embedding_tensor


    

    #     print("modified text prediction::", is_modified_toxic)

    #     print("group modified text prediction::", is_group_modified_toxic)

    #     if is_modified_toxic != is_origin_toxic:
    #         changed_idx_ls.append(idx)


def get_group_id_ls(X_group):
    unique_ids = np.unique(X_group)


    sample_ids_ls = []
    for idx in unique_ids:
        sample_ids = np.nonzero(X_group == idx).reshape(-1)
        sample_ids_ls.append(sample_ids)


    return sample_ids_ls


def select_random_ids(args, save_dir,len1, len2, count):
    if args.train:
               
        # selected_idx0 = select_samples_by_clustering(X0, group_idx_ls0, count = count)
        # selected_idx1 = select_samples_by_clustering(X1, group_idx_ls1, count = count)
        if os.path.exists(os.path.join(save_dir, "selected_idx0")):
            selected_idx0 = load_objs(os.path.join(save_dir, "selected_idx0"))
        else:
            new_set = True
            selected_idx0 = np.random.choice(len1, count, replace=False)
        
        if os.path.exists(os.path.join(save_dir, "selected_idx1")):
            selected_idx1 = load_objs(os.path.join(save_dir, "selected_idx1"))
        else:
            selected_idx1 = np.random.choice(len2, count, replace=False)
        print("idx0::", selected_idx0)
        print("idx1::", selected_idx1)
        save_objs(selected_idx0, os.path.join(save_dir, "selected_idx0"))
        save_objs(selected_idx1, os.path.join(save_dir, "selected_idx1"))
    else:
        selected_idx0 = load_objs(os.path.join(save_dir, "selected_idx0"))
        selected_idx1 = load_objs(os.path.join(save_dir, "selected_idx1"))
        print("idx0::", selected_idx0)
        print("idx1::", selected_idx1)
    return selected_idx0, selected_idx1

def get_word_pair_sims(word_embedding_ls1, word_embedding_ls2):
    word_pair_inner_prod = torch.sum(word_embedding_ls1.unsqueeze(1)*word_embedding_ls2.unsqueeze(0),dim=-1)
    word_embedding_norm1 = torch.norm(word_embedding_ls1, dim = -1).unsqueeze(-1)
    word_embedding_norm2 = torch.norm(word_embedding_ls2, dim = -1).unsqueeze(-1)
    word_embedding_norm_prod = torch.sum(word_embedding_norm1.unsqueeze(1)*word_embedding_norm2.unsqueeze(0))
    word_embedding_cosin_sim = word_pair_inner_prod/word_embedding_norm_prod
    print()
    return word_embedding_cosin_sim

# Mean Pooling - Take attention mask into account for correct averaging
# From Huggingface
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_sentence_embeddings(dataset):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Tokenize sentences
    encoded_input = tokenizer(dataset, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def cluster_embeddings(embeddings):
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
    return torch.stack([torch.tensor(kmeans.labels_) == 0,
                        torch.tensor(kmeans.labels_) == 1], dim=1)
                        # torch.tensor(kmeans.labels_) == 2,
                        # torch.tensor(kmeans.labels_) == 3], dim=1)

def main(args):

    # api_key = retrieve_api_key_file(args.api_key_file)

    # openai.api_key = api_key

    experiment_name = 'CivilComments'
    data_dir = Path(args.data_dir)


    # Setting so that the plots look normal even when using dark-reader
    sn.set_style("whitegrid")
    sn.set_context(context="paper", font_scale=2)
    pal = sn.color_palette("Set1")

    # Setting up specifics for plotting + saving
    save_figures = True
    add_legend = False
    add_title = False
    add_axis = False
    save_parms = {'format': 'pdf','bbox_inches':'tight', 'pad_inches':0}
    line_parms = {'linewidth':4, 'color':'k'}
    if save_figures:
        figure_dir = Path('.') / 'figures' / experiment_name  # saves all figures in a figure directory in the local directory
        if not figure_dir.exists():
            figure_dir.mkdir(parents=True)


    entire_notebook_start_time = time()

    # data hyperparameters
    max_features = 500
    # since we have *many features* now, we limit the max number of features features we search over
    max_k_shift = 100

    # Choose one of the comment sections below to uncomment, and then run the notebook with that uncommented
    # to recreate that experiment!

    # uncomment below for Explain('female' -> 'male')
    # is_subdemographic = False
    # training_frac = 0.2  # used to balance the sample numbers across experiments
    # demo1 = 'female'
    # demo2 = 'male'
    #

    # # uncomment below for Explain('F_0' -> 'F_1')
    # is_subdemographic = True
    # training_frac = 1  # used to balance the sample numbers across experiments
    # demo_base = 'female'
    ##

    # # uncomment below for Explain('M_0' -> 'M_1')
    is_subdemographic = True
    training_frac = 1  # used to balance the sample numbers across experiments
    demo_base = 'male'
    ##

    if is_subdemographic:
        # demo1 = demo_base + '_toxic'  # nontoxic
        # demo2 = demo_base + '_nontoxic'     # toxic
        demo1 = 'nontoxic'  # nontoxic
        demo2 = 'toxic'     # toxic
    else:
        demo1 = "male"
        demo2 = "female"

    from wilds import get_dataset

    group_names = np.array(['male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white',
            'identity_any',
            'severe_toxicity',
            'obscene',
            'threat',
            'insult',
            'identity_attack',
            'sexual_explicit'])

    dataset = get_dataset(dataset='civilcomments', unlabeled=False, root_dir=str(data_dir), download=True)

    train = dataset.get_subset('train', frac=training_frac)
    # train = dataset.get_subset('train')

    demographics_data = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}
    demographics_groups = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}
    demographics_label = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}


    # splitting the training data up into the demographic groups
    all_data_arr = np.array(train, dtype=object)  # moving to np since indexing can be quirky in pytorch
    for metadata_col, corresponding_key in enumerate(demographics_data):
        rows_in_demographic = train.metadata_array[:, metadata_col] == 1
        demographics_data[corresponding_key] = all_data_arr[rows_in_demographic][:, 0]
        demographics_groups[corresponding_key] = train.metadata_array[rows_in_demographic]
        demographics_label[corresponding_key] = np.array([item.item() for item in all_data_arr[rows_in_demographic][:, 1]])

    if is_subdemographic:
        # adding base_nontoxic and base_toxic to the demographics data and labels
        demographics_data[demo1] = demographics_data[demo_base][demographics_label[demo_base] == 0]
        demographics_groups[demo1] = demographics_groups[demo_base][demographics_label[demo_base] == 0]
        demographics_label[demo1] = demographics_label[demo_base][demographics_label[demo_base] == 0]

        demographics_data[demo2] = demographics_data[demo_base][demographics_label[demo_base] == 1]
        demographics_groups[demo2] = demographics_groups[demo_base][demographics_label[demo_base] == 1]
        demographics_label[demo2] = demographics_label[demo_base][demographics_label[demo_base] == 1]

    # print(demographics_groups)
    
    ngram = 1  # used for setting the ngram range when building the ngram bag of words

    save_dir = os.path.join(data_dir, demo1 + "_VS_" + demo2 + "_" + args.method)
    os.makedirs(save_dir, exist_ok= True)

    training_explanations = args.train
    other_data = {}
    other_groups = {}
    other_labels = {}
    full_data = {}
    full_groups = {}
    full_labels = {}

    # og_X0 = demographics_data[demo1]

    # og_X1 = demographics_data[demo2]

    # X0_group = demographics_groups[demo1]
    
    # X1_group = demographics_groups[demo2]

    # average_X0_group = torch.mean(X0_group.float(), dim=0)
    # average_X1_group = torch.mean(X1_group.float(), dim=0)

    # sorted_feats =  torch.argsort(torch.abs(average_X0_group - 0.5))

    # selected_group_ids = sorted_feats[0:1]

    # X0_group = X0_group[:,selected_group_ids]
    # X1_group = X1_group[:,selected_group_ids]


    # og_X_both = np.concatenate((og_X0, og_X1))

    # ngram_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),
    #                                    stop_words='english', max_features=max_features)
    # vectorized_data = ngram_vectorizer.fit_transform(og_X_both)

    # X0, X1 = vectorized_data[:og_X0.shape[0]], vectorized_data[og_X0.shape[0]:] 

    # X0 = X0.toarray().astype(float)
    # X1 = X1.toarray().astype(float)

    # group_idx_ls0 = get_group_id_ls(X0_group)
    # group_idx_ls1 = get_group_id_ls(X1_group)



    count = 200
    new_set = False

      
    
    
    if not args.method == "dice":
        selected_idx0, selected_idx1 = select_random_ids(args, save_dir, len(demographics_data[demo1]), len(demographics_data[demo2]), count)
        downsample_data(selected_idx0, demographics_data, demographics_groups, demographics_label, other_data, other_groups, other_labels, demo1, full_data, full_groups, full_labels)
        downsample_data(selected_idx1, demographics_data, demographics_groups, demographics_label, other_data, other_groups, other_labels, demo2, full_data, full_groups, full_labels)
        X0, X1, feature_name_ls, origin_X0, origin_X1, X0_group, X1_group = get_demographic_counts(is_subdemographic, save_dir, demo1, demo2, demographics_data,demographics_groups,
                                                        ngram, max_features, equalize_sizes=False)
        print(X0.shape, X1.shape, X0_group.shape, X1_group.shape)
    else:
        
        full_X0, full_X1, feature_name_ls, full_origin_X0, full_origin_X1, full_X0_group, full_X1_group = get_demographic_counts(is_subdemographic, save_dir, demo1, demo2, demographics_data, demographics_groups,
                                                        ngram, max_features, equalize_sizes=True, sort_word=False)
        
        selected_idx0, selected_idx1 = select_random_ids(args, save_dir, len(full_X0), len(full_X1), count)
        X0 = full_X0
        origin_X0 = full_origin_X0
        origin_X1 = full_origin_X1
        X0_group = full_X0_group
        X1_group = full_X1_group
        
        # X0 = full_X0[selected_idx0]
        # # X1 = full_X1[selected_idx1]
        # origin_X0 = full_origin_X0[selected_idx0]
        # # origin_X1 = full_origin_X1[selected_idx1]
        # X0_group = full_X0_group[selected_idx0]
        # # X1_group = full_X1_group[selected_idx1]
        print(X0.shape, X0_group.shape)
        save_objs(feature_name_ls, os.path.join(save_dir, "feature_name_ls"))

    average_X0_group = torch.mean(X0_group.float(), dim=0)
    average_X0_group = average_X0_group[0:15]
    
    sorted_feats =  torch.argsort(torch.abs(average_X0_group - 0.5))
    print(average_X0_group[sorted_feats])

    
    # if is_subdemographic:
    sorted_feats =  torch.argsort(torch.abs(average_X0_group - 0.5))
    print(group_names[sorted_feats])
    # selected_group_ids = sorted_feats[1:2]
    selected_group_ids = sorted_feats[0:1]
    X0_group = X0_group[:, selected_group_ids] #(torch.sum(X0_group[:,selected_group_ids], dim=1, keepdim=True) > 0).float()
    X1_group = X1_group[:, selected_group_ids] #(torch.sum(X1_group[:,selected_group_ids], dim=1, keepdim=True) > 0).float()

    # Use clustering based groups
    if args.no_groups:
        X0_embeddings = extract_sentence_embeddings(list(origin_X0))
        X1_embeddings = extract_sentence_embeddings(list(origin_X1))
        all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
        X0_group = all_groups[:len(X0_embeddings)]
        X1_group = all_groups[len(X0_embeddings):]
        print(X0_group.shape, X1_group.shape)

    print("group counts:", torch.sum(X0_group, dim=0) / X0_group.shape[0], torch.sum(X1_group, dim=0) / X1_group.shape[0])
    
    
    # else:
    #     X0_group = X0_group[:,sorted_feats[1:2]]
    #     X1_group = X1_group[:,sorted_feats[1:2]]
    #     selected_group_ids = sorted_feats[1:2]
    
    
    if not args.method == 'dice':
        og_X0 = demographics_data[demo1]
        og_X1 = demographics_data[demo2]
        average_X1_group = torch.mean(X1_group.float(), dim=0)
        average_X1_group = average_X1_group[0:15]
        # X1_group = X1_group[:,selected_group_ids]
        if new_set:
            og_X0_embeddings = get_toxic_predictions_for_all(og_X0.tolist())
            og_X1_embeddings = get_toxic_predictions_for_all(og_X1.tolist())

            save_objs(og_X0_embeddings, os.path.join(save_dir, "X0_embeddings"))
            save_objs(og_X1_embeddings, os.path.join(save_dir, "X1_embeddings"))
    # other_X0 = transform_sentences_to_vectors(other_data[demo1], ngram_vectorizer)
    # other_X1 = transform_sentences_to_vectors(other_data[demo2], ngram_vectorizer)


    if training_explanations:
        if not args.method == "dice":
            for i in range(3):
                compute_explanations(i, args, X0, X1, X0_group, X1_group, save_dir, feature_name_ls, max_features, kmeans_cluster_count=args.kmeans_cluster_count)
        
        else:
            # X1_group = X1_group[:,selected_group_ids]
            print("X0 shape:", X0.shape)
            save_full_X0_group = X0_group.clone()
            save_full_X1_group = X1_group.clone()

            for i in range(3):
                full_X0_group = save_full_X0_group.clone()
                full_X1_group = save_full_X1_group.clone()
                net_regular = train_classifying_source_target_model_regular(full_X0,full_X1, full_X0_group, full_X1_group)
                net_group = train_classifying_source_target_model_group(full_X0,full_X1, full_X0_group, full_X1_group)
                X1, selected_sample_ids1 = select_target_samples(count, net_regular, net_group, full_X1)
                X1_group = full_X1_group[selected_sample_ids1, :]
                save_objs(selected_sample_ids1, os.path.join(save_dir, f"selected_sample_ids1_{i}"))

                compute_explanations_for_dice(i, X0, X1, selected_idx0, full_X0, full_X1, X0_group, X1_group, full_X1_group, save_dir, feature_name_ls, net_regular, group_method=False)
                compute_explanations_for_dice(i, X0, X1, selected_idx0, full_X0, full_X1, X0_group, X1_group, full_X1_group, save_dir, feature_name_ls, net_group, group_method=True)

                if new_set:
                    og_X0 = full_data[demo1][selected_idx0]
                    og_X1 = full_data[demo2][selected_idx1]
                    og_X0_embeddings = get_toxic_predictions_for_all(og_X0.tolist())
                    og_X1_embeddings = get_toxic_predictions_for_all(og_X1.tolist())

                    save_objs(og_X0_embeddings, os.path.join(save_dir, f"X0_embeddings_{i}"))
                    save_objs(og_X1_embeddings, os.path.join(save_dir, f"X1_embeddings_{i}"))
        
        
        # save_objs(ngram_vectorizer, os.path.join(save_dir, "vectorizer"))
        # save_objs(maxabs_transformer, os.path.join(save_dir, "maxabs_transformer"))

        
    else:

        # maxabs_transformer = load_objs(os.path.join(save_dir, "maxabs_transformer"))
        og_X_both = np.concatenate((origin_X0, origin_X1))

        ngram_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),
                                           stop_words='english')
        # ngram_vectorizer = load_objs(os.path.join(save_dir, "vectorizer"))

        all_topic_word_ls = []
        all_topic_word_ls.extend(train.dataset._identity_vars)
        all_topic_word_ls.extend(train.dataset._auxiliary_vars)
        # all_feature_word_embeddings = get_word_embeddings_for_all(list(ngram_vectorizer.get_feature_names_out()))
        # all_topic_word_embeddings = get_word_embeddings_for_all(all_topic_word_ls)

        gender_tokenizer = AutoTokenizer.from_pretrained("padmajabfrl/Gender-Classification")
        gender_model = AutoModelForSequenceClassification.from_pretrained("padmajabfrl/Gender-Classification")
        # selected_topic_word_embeddings = all_topic_word_embeddings[selected_group_ids]

        # all_non_topic_word_embeddings = get_word_embeddings_for_all_negated(all_topic_word_ls)

        # all_topic_word_embeddings = load_objs(os.path.join(save_dir, "all_topic_word_embeddings"))
        # all_feature_word_embeddings = load_objs(os.path.join(save_dir, "all_feature_word_embeddings"))
        # word_embedding_cosin_sim = get_word_pair_sims(torch.cat([all_topic_word_embeddings, all_non_topic_word_embeddings], 0), all_feature_word_embeddings)
        # save_objs(word_embedding_cosin_sim, os.path.join(save_dir, "word_embedding_cosin_sim"))

        # selected_topic_word_embeddings = all_topic_word_embeddings[selected_group_ids[0]]

        
        # og_X0_embeddings = get_word_embeddings_for_all(og_X0.tolist())
        # og_X1_embeddings = get_word_embeddings_for_all(og_X1.tolist())
        # save_objs(og_X0_embeddings, os.path.join(save_dir, "X0_embeddings"))
        # save_objs(og_X1_embeddings, os.path.join(save_dir, "X1_embeddings"))
        # og_X_both = np.concatenate((og_X0, og_X1))
        vectorized_data = ngram_vectorizer.fit_transform(og_X_both).toarray()
        # vectorized_data = ngram_vectorizer.transform(og_X_both)
        X0, X1 = vectorized_data[:origin_X0.shape[0]], vectorized_data[origin_X0.shape[0]:] 
        X_combined = np.concatenate([X0, X1], axis=0)
        sel = SelectKBest(chi2, k=50).fit(X_combined, np.concatenate([np.zeros(X0.shape[0]), np.ones(X1.shape[0])], axis=0))
        feature_name_ls = ngram_vectorizer.get_feature_names_out()
        feature_name_ls = list(feature_name_ls[sel.get_support()])
        X0 = X0[:, sel.get_support()]
        X1 = X1[:, sel.get_support()]

        normalized_X0 = X0#maxabs_transformer.transform(X0)
        normalized_X1 = X1#maxabs_transformer.transform(X1)

        reg_total = []
        reg_worst = []
        reg_flip = []
        reg_feas = []
        group_total = []
        group_worst = []
        group_flip = []
        group_feas = []

        normalized_X0 = normalized_X0.astype(float)
        normalized_X1 = normalized_X1.astype(float)

        print(origin_X0.shape)

        save_normalized_X0 = normalized_X0.copy()
        save_normalized_X1 = normalized_X1.copy()
        # save_full_origin_X0 = full_origin_X0.copy()
        # save_full_origin_X1 = full_origin_X1.copy()
        save_X0 = X0.copy()
        save_X1 = X1.copy()
        save_X0_group = X0_group.clone()
        save_X1_group = X1_group.clone()
        save_origin_X0 = origin_X0.copy()
        save_origin_X1 = origin_X1.copy()
        if args.no_groups:
            save_X1_embeddings = X1_embeddings.numpy().copy()
        runs = 3 #if args.method == "dice" else 1
        for i in range(runs):
            normalized_X0 = save_normalized_X0.copy()
            normalized_X1 = save_normalized_X1.copy()
            # full_origin_X0 = save_full_origin_X0.copy()
            # full_origin_X1 = save_full_origin_X1.copy()
            X0 = save_X0.copy()
            X1 = save_X1.copy()
            X0_group = save_X0_group.clone()
            X1_group = save_X1_group.clone()
            origin_X0 = save_origin_X0.copy()
            origin_X1 = save_origin_X1.copy()
            if args.no_groups:
                X1_embeddings = save_X1_embeddings.copy()
            if args.method == "dice":
                selected_idx1 = load_objs(os.path.join(save_dir, f"selected_sample_ids1_{i}"))
                normalized_X0 = normalized_X0[selected_idx0]
                normalized_X1 = normalized_X1[selected_idx1]
                full_origin_X0 = origin_X0[selected_idx0]
                full_origin_X1 = origin_X1[selected_idx1]
                X0 = X0[selected_idx0]
                X1 = X1[selected_idx1]
                if args.no_groups:
                    X1_embeddings = X1_embeddings[selected_idx1]
                X0_group = X0_group[selected_idx0]
                X1_group = X1_group[selected_idx1]
                origin_X0 = origin_X0[selected_idx0]
                origin_X1 = full_origin_X1

            with open(os.path.join(save_dir, f"origin_x_s_{i}"), "rb") as f:
                origin_x_s = pickle.load(f)

            with open(os.path.join(save_dir, f"group_x_s_{i}"), "rb") as f:
                group_x_s = pickle.load(f)

            if False and os.path.exists(os.path.join(save_dir, f"X0_embeddings_{i}")):
                og_X0_embeddings = load_objs(os.path.join(save_dir, f"X0_embeddings_{i}"))
            else:
                og_X0_embeddings = get_toxic_predictions_for_all(origin_X0.tolist())
                save_objs(og_X0_embeddings, os.path.join(save_dir, f"X0_embeddings_{i}"))
            
            if False and os.path.exists(os.path.join(save_dir, f"X1_embeddings_{i}")):
                og_X1_embeddings = load_objs(os.path.join(save_dir, f"X1_embeddings_{i}"))
            else:
                og_X1_embeddings = get_toxic_predictions_for_all(origin_X1.tolist())
                save_objs(og_X1_embeddings, os.path.join(save_dir, f"X1_embeddings_{i}"))
            
            # og_X1_embeddings = load_objs(os.path.join(save_dir, "X1_embeddings"))
            origin_x_s = origin_x_s.round(0)
            if type(origin_x_s) != np.ndarray:
                origin_x_s = origin_x_s.to_numpy()
            print(normalized_X0.shape, origin_x_s.shape, X0_group.numpy().shape)
            print("Feature space PE:")
            total, worst = group_percent_explained(
                    normalized_X0,
                    origin_x_s,
                    normalized_X1,
                    X0_group.numpy(),
                    X1_group.numpy(),
                    feature_name_ls)

            group_x_s = group_x_s.round(0)
            if type(group_x_s) != np.ndarray:
                group_x_s = group_x_s.to_numpy()
            g_total, g_worst = group_percent_explained(
                    normalized_X0,
                    group_x_s,
                    normalized_X1,
                    X0_group.numpy(),
                    X1_group.numpy(),
                    feature_name_ls)

            reg_total.append(total)
            reg_worst.append(worst)
            group_total.append(g_total)
            group_worst.append(g_worst)

            X0 = X0
            X1 = X1
            if False and os.path.exists(os.path.join(save_dir, f"modified_X0_embedding_tensor_{i}")) and os.path.exists(os.path.join(save_dir, f"group_modified_X0_embedding_tensor_{i}")):
                modified_X0_embedding_tensor = load_objs(os.path.join(save_dir,
                    f"modified_X0_embedding_tensor_{i}"))
                group_modified_X0_embedding_tensor = load_objs(os.path.join(save_dir,
                    f"group_modified_X0_embedding_tensor_{i}"))
            else:
                modified_X0_embedding_tensor, group_modified_X0_embedding_tensor = edit_text_based_on_explanation(X0, og_X0_embeddings, origin_x_s, group_x_s, origin_X0, feature_name_ls, X0_group[:,0], gender_tokenizer, gender_model)
                save_objs(modified_X0_embedding_tensor, os.path.join(save_dir, f"modified_X0_embedding_tensor_{i}"))
                save_objs(group_modified_X0_embedding_tensor, os.path.join(save_dir, f"group_modified_X0_embedding_tensor_{i}"))


            # print("Embedding PE:")
            # group_percent_explained(og_X0_embeddings.numpy(), modified_X0_embedding_tensor.numpy(), og_X1_embeddings.numpy(), X0_group.numpy(), X1_group.numpy(), feature_name_ls)
            # group_percent_explained(og_X0_embeddings.numpy(), group_modified_X0_embedding_tensor.numpy(), og_X1_embeddings.numpy(), X0_group.numpy(), X1_group.numpy(), feature_name_ls)

            X0_toxic = np.sum((og_X0_embeddings.numpy() > 0.5), axis=1) > 0
            modified_X0_toxic = np.sum((modified_X0_embedding_tensor.numpy() > 0.5), axis=1) > 0
            group_modified_X0_toxic = (group_modified_X0_embedding_tensor.numpy()[:, 0] > 0.5)

            def edit_text(sentence, words, diff):
                edits = {w: diff[i] for i, w in enumerate(words)}
                new_sentence = []
                for w, e in edits.items():
                    if e > 0:
                        new_sentence += [w] * int(e)

                for w in sentence.lower().split():
                    if w in edits and edits[w] < 0:
                        edits[w] += 1
                        continue
                    new_sentence.append(w)
                return " ".join(new_sentence)

            # Assign group labels to transformed source
            if args.no_groups:
                print(np.sum(origin_x_s - normalized_X0))
                neigh = KNeighborsClassifier(n_neighbors=1).fit(X1_embeddings, np.arange(len(X1_embeddings)))
                x_s_counterfactual_text = [edit_text(s, feature_name_ls, diff) for s, diff in zip(origin_X0, origin_x_s - normalized_X0)]
                x_s_group_counterfactual_text = [edit_text(s, feature_name_ls, diff) for s, diff in zip(origin_X0, group_x_s - normalized_X0)]
                x_s_groups = X1_group.numpy()[neigh.predict(extract_sentence_embeddings(x_s_counterfactual_text))]
                group_x_s_groups = X1_group.numpy()[neigh.predict(extract_sentence_embeddings(x_s_group_counterfactual_text))]
            else:
                neigh = KNeighborsClassifier(n_neighbors=1).fit(normalized_X1, np.arange(len(normalized_X1)))
                x_s_groups = X1_group.numpy()[neigh.predict(origin_x_s)]
                group_x_s_groups = X1_group.numpy()[neigh.predict(group_x_s)]
            feas = np.sum(np.sum(X0_group.numpy() == x_s_groups, axis=1) == X0_group.numpy().shape[1]) / X0_group.numpy().shape[0]
            g_feas = np.sum(np.sum(X0_group.numpy() == group_x_s_groups, axis=1) == X0_group.numpy().shape[1]) / X0_group.numpy().shape[0]
            print(feas, g_feas)
            reg_feas.append(feas)
            group_feas.append(g_feas)

            print("Initial percent toxic:", 100 * np.sum(X0_toxic) / X0_toxic.shape[0])
            flipped = np.logical_and(~X0_toxic, modified_X0_toxic)
            group_flipped = np.logical_and(~X0_toxic, group_modified_X0_toxic)
            print("Percent Flipped:", 100 * np.sum(flipped) / flipped.shape[0])
            print("Percent Flipped:", 100 * np.sum(group_flipped) / group_flipped.shape[0])
            reg_flip.append(np.sum(flipped) / flipped.shape[0])
            group_flip.append(np.sum(group_flipped) / group_flipped.shape[0])

        print(np.mean(reg_total), np.std(reg_total))
        print(np.mean(reg_worst), np.std(reg_worst))
        print(np.mean(reg_flip), np.std(reg_flip))
        print(np.mean(reg_feas), np.std(reg_feas))
        print(np.mean(group_total), np.std(group_total))
        print(np.mean(group_worst), np.std(group_worst))
        print(np.mean(group_flip), np.std(group_flip))
        print(np.mean(group_feas), np.std(group_feas))

        centroids = load_objs(os.path.join(save_dir, "centroids"))
        shifts = load_objs(os.path.join(save_dir, "shifts"))

        g_centroids = load_objs(os.path.join(save_dir, "group_centroids"))
        g_shifts = load_objs(os.path.join(save_dir, "group_shifts")).round(0)

        diffs = np.abs(shifts - g_shifts)
        sort_idx = np.argsort(-diffs, axis=1)
        features = load_objs(os.path.join(save_dir, "feature_name_ls"))
        j = 0
        for centroid, shift in zip(centroids, shifts):
            print([f"{features[i]}: {centroid[i].round(1)}" for i in np.nonzero(centroid.round(1))[0]])
            for i in sort_idx[j][:50]:
                print(f"Shift in {features[i]} by {shift[i].round(1)}.")
            j += 1

        print()
        j = 0
        for centroid, shift in zip(g_centroids, g_shifts):
            print([f"{features[i]}: {centroid[i].round(1)}" for i in np.nonzero(centroid.round(1))[0]])
            for i in sort_idx[j][:50]:
                print(f"Shift in {features[i]} by {shift[i].round(1)}. ({diffs[j][i]})")
            j += 1


        with open(os.path.join(save_dir, f"origin_x_s_0"), "rb") as f:
            origin_x_s = pickle.load(f)

        with open(os.path.join(save_dir, f"group_x_s_0"), "rb") as f:
            group_x_s = pickle.load(f)
        neigh = KNeighborsClassifier(n_neighbors=1).fit(centroids, np.arange(len(centroids)))
        neigh_s2t = KNeighborsClassifier(n_neighbors=1).fit(X1, np.arange(len(X1)))
        
        cluster_asgn = neigh.predict(X0)
        target_asgn = neigh_s2t.predict(origin_x_s)
        target_asgn_g = neigh_s2t.predict(group_x_s)
        
        cluster_idx = 1
        cluster3_source = X0[cluster_asgn == cluster_idx]
        cluster3_target = X1[target_asgn][cluster_asgn == cluster_idx]
        cluster3_target_g = X1[target_asgn_g][cluster_asgn == cluster_idx]
        
        cluster3_transformed = X0[cluster_asgn == cluster_idx] + shifts[cluster_idx]
        cluster3_transformed_g = X0[cluster_asgn == cluster_idx] + g_shifts[cluster_idx]
        print("Source features:", [f"{features[sort_idx[cluster_idx][i]]}: {s:.2f}" for i, s in enumerate(cluster3_source[0][sort_idx[cluster_idx]]) if s != 0])
        print("Transformed source:", [f"{features[sort_idx[cluster_idx][i]]}: {s:.2f}" for i, s in enumerate(cluster3_transformed[0][sort_idx[cluster_idx]]) if s != 0])
        print(origin_X0[cluster_asgn == cluster_idx][0])
        print()
        print("Group transformed:", [f"{features[sort_idx[cluster_idx][i]]}: {s:.2f}" for i, s in enumerate(cluster3_transformed_g[0][sort_idx[cluster_idx]]) if s != 0])
        print()
        print("Nearest target:", [f"{features[sort_idx[cluster_idx][i]]}: {s:.2f}" for i, s in enumerate(cluster3_target[0][sort_idx[cluster_idx]]) if s != 0])
        print("Group nearest target:", [f"{features[sort_idx[cluster_idx][i]]}: {s:.2f}" for i, s in enumerate(cluster3_target_g[0][sort_idx[cluster_idx]]) if s != 0])
        print(origin_X0[cluster_asgn == cluster_idx][0])
        print(origin_X0[cluster_asgn == cluster_idx][1])
        print(origin_X0[cluster_asgn == cluster_idx][2])
        print(origin_X1[target_asgn][cluster_asgn == cluster_idx][0])
        print(origin_X1[target_asgn][cluster_asgn == cluster_idx][1])
        print(origin_X1[target_asgn][cluster_asgn == cluster_idx][2])
        

        # edit_text_based_on_explanation_and_predict_labels(X0, origin_x_s, group_x_s, origin_X0, list(ngram_vectorizer.get_feature_names_out()), "toxic" in demo1)
        # edit_text_based_on_explanation(X0, origin_x_s, group_x_s, origin_X0, list(ngram_vectorizer.get_feature_names_out()), selected_topic_word_embeddings, X0_group[:,0])


        # group_x_s = (group_x_s + 0.5).astype(np.int32)
        # group_percent_explained(X0, group_x_s, X1, X0_group.numpy(), X1_group.numpy(), list(ngram_vectorizer.get_feature_names_out()))
        # s_len_ls = []
        # for idx in range(len(origin_X0)):
        #     s_len = len(origin_X0[idx])
        #     s_len_ls.append(s_len)

        # s_len_ls_tensor = torch.tensor(s_len_ls)
        # sorted_id_ls = torch.argsort(s_len_ls_tensor, descending=False).numpy()
        # selected_sorted_id_ls = sorted_id_ls[0:100]
        # origin_X0 = origin_X0[selected_sorted_id_ls]
        # origin_x_s = origin_x_s[selected_sorted_id_ls]
        # X0 = X0[selected_sorted_id_ls]

        # X0 = maxabs_transformer.inverse_transform(X0)
        # X1 = maxabs_transformer.inverse_transform(X1)
        # origin_x_s = maxabs_transformer.inverse_transform(origin_x_s)
        # group_x_s = maxabs_transformer.inverse_transform(group_x_s)

        # start_sample_id = 0
        # end_sample_id = 2000
        # origin_X0 = origin_X0[start_sample_id:end_sample_id]
        # origin_x_s = origin_x_s[start_sample_id:end_sample_id]
        # group_x_s = group_x_s[start_sample_id:end_sample_id]
        # X0 = X0[start_sample_id:end_sample_id]
        
        # changed_pred_count2 = edit_text_based_on_explanation(X0, group_x_s, origin_X0, list(ngram_vectorizer.get_feature_names_out()))
        # print("result::")

        # print(changed_pred_count1, changed_pred_count2)
        # orig_other_x_s = transform_samples(other_X0, X0, origin_x_s)
        # group_other_x_s = transform_samples(other_X0, X0, group_x_s)
        # edit_text_based_on_explanation(other_X0, orig_other_x_s, group_other_x_s, other_data[demo1], list(ngram_vectorizer.get_feature_names_out()))
        # compute_transformed_diff(orig_other_x_s, other_X0, other_data[demo1], ngram_vectorizer)
        # print(changed_pred_count1, changed_pred_count2)

        # group_other_x_s = transform_samples(other_X0, X0, group_x_s)
        # compute_transformed_diff(group_other_x_s, other_X0, other_data[demo1], ngram_vectorizer)

            # pickle.dump(x_s, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
