import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, OrdinalEncoder
import torch
from pathlib import Path
import csv
from robustness.tools.breeds_helpers import ClassHierarchy, BreedsDatasetGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.cluster import KMeans, FeatureAgglomeration
import torch.nn.functional as F
from wilds import get_dataset
import os

from src.optimal_transport import group_mean_shift_transport, group_kmeans_shift_transport, transform_samples_kmeans, transform_samples, group_feature_transport
from src.distance import group_percent_explained, W2_dist
from src.training import regular_training, dro_training
from src.logistic_regression import PTLogisticRegression, PTNN, FFNetwork
from src.training import regular_training
from src.cf_transport import get_dice_transformed, get_closest_target

# import matplotlib.pyplot as plt
from sklearn import preprocessing
# import tikzplotlib

def get_demographic_counts(demo1, demo2, demographics_data,demographics_groups,
                           ngram, max_features, equalize_sizes=False):
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
    ngram_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),
                                       stop_words='english') #, max_features=max_features)
    og_X_both = np.concatenate((og_X0, og_X1))
    
    vectorized_data = ngram_vectorizer.fit_transform(og_X_both).toarray()

    X0, X1 = vectorized_data[:og_X0.shape[0]].astype(float), vectorized_data[og_X0.shape[0]:].astype(float)
    feature_name_ls = ngram_vectorizer.get_feature_names_out()

    X_combined = np.concatenate([X0, X1], axis=0)
    sel = SelectKBest(chi2, k=50).fit(X_combined, np.concatenate([np.zeros(X0.shape[0]), np.ones(X1.shape[0])], axis=0))
    feature_name_ls = list(feature_name_ls[sel.get_support()])
    X0 = X0[:, sel.get_support()]
    X1 = X1[:, sel.get_support()]
    
    return X0, X1, feature_name_ls, og_X0, og_X1, og_X0_group, og_X1_group

def featurize_nlp(data):
    all_source = []
    all_target = []
    for d in data:
        if d[1].item() == 0:
            all_source.append(d[0])
        else:
            all_target.append(d[0])
    all_source = np.array(all_source)
    all_target = np.array(all_target)

    # equalize sizes
    rng = np.random.RandomState(42)
    subsample_idxs = rng.choice(len(all_target), replace=False, size=1000)
    all_target = all_target[subsample_idxs]
    subsample_idxs = rng.choice(len(all_source), replace=False, size=1000)
    all_source = all_source[subsample_idxs]

    ngram_vectorizer = CountVectorizer(ngram_range=(1,1),
                                       stop_words='english')
    all_data = np.concatenate((all_source, all_target))
    vectorized_data = ngram_vectorizer.fit_transform(all_data).toarray()

    X0, X1 = vectorized_data[:len(all_source)].astype(float), vectorized_data[len(all_source):].astype(float)
    feature_name_ls = ngram_vectorizer.get_feature_names_out()

    X_combined = np.concatenate([X0, X1], axis=0)
    # sel = VarianceThreshold(threshold=(.9 * (1 - .9))).fit(X_combined)
    # feature_name_ls = list(feature_name_ls[sel.get_support()])
    # print("Resulting features:", len(feature_name_ls))
    sel = SelectKBest(chi2, k=50).fit(X_combined, np.concatenate([np.zeros(X0.shape[0]), np.ones(X1.shape[0])], axis=0))
    feature_name_ls = list(feature_name_ls[sel.get_support()])
    X0 = X0[:, sel.get_support()]
    X1 = X1[:, sel.get_support()]
    
    return X0, X1, feature_name_ls, all_source, all_target


def load_nlp():
    data_dir = Path('../data/nlp')
    demo_base = 'male'
    demo1 = 'nontoxic'  # nontoxic
    demo2 = 'toxic'     # toxic
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
    train = dataset.get_subset('train', frac=1)

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

    # adding base_nontoxic and base_toxic to the demographics data and labels
    demographics_data[demo1] = demographics_data[demo_base][demographics_label[demo_base] == 0]
    demographics_groups[demo1] = demographics_groups[demo_base][demographics_label[demo_base] == 0]
    demographics_label[demo1] = demographics_label[demo_base][demographics_label[demo_base] == 0]

    demographics_data[demo2] = demographics_data[demo_base][demographics_label[demo_base] == 1]
    demographics_groups[demo2] = demographics_groups[demo_base][demographics_label[demo_base] == 1]
    demographics_label[demo2] = demographics_label[demo_base][demographics_label[demo_base] == 1]

    bow_source, bow_target, feature_names, source, target, source_group, target_group = get_demographic_counts(demo1, demo2, demographics_data,demographics_groups,
                                                    1, 500, equalize_sizes=True)

    # bow_source, bow_target, feature_names, source, target = featurize_nlp(train)

    id_scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    return ["int"] * bow_source.shape[1], id_scaler, source, target, bow_source, bow_target, source_group.numpy(), target_group.numpy(), feature_names

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
    groups = []
    for c in range(len(np.unique(kmeans.labels_))):
        groups.append(kmeans.labels_ == c)
        print(f"Samples in cluster {c}: {np.sum(kmeans.labels_ == c)}")

    return np.stack(groups, axis=1)


def load_imagenet():
    df = pd.read_json("../data/imagenetx/imagenet_x_val_multi_factor.jsonl", lines=True)
    hier = ClassHierarchy("../data/breeds/")
    level = 3 # Could be any number smaller than max level
    superclasses = hier.get_nodes_at_level(level)
    
    DG = BreedsDatasetGenerator("../data/breeds/")
    ret = DG.get_superclasses(level=4,
        Nsubclasses=6,
        split="bad",
        ancestor="n01861778",
        balanced=True)
    superclasses, subclass_split, label_map = ret
    def flatlist(l):
        return [item for sublist in l for item in sublist]


    source_classes = []
    target_classes = []
    for i in flatlist(subclass_split[0][1:3]):
        source_classes.append(i)
    for i in flatlist(subclass_split[1][1:3]):
        target_classes.append(i)

    source_df = df[df["class"].isin(source_classes)]
    target_df = df[df["class"].isin(target_classes)]
    source_files = source_df["file_name"].to_list()
    target_files = target_df["file_name"].to_list()
    source_labels = source_df["class"].isin(subclass_split[0][1]).to_numpy().astype(float)
    target_labels = target_df["class"].isin(subclass_split[1][1]).to_numpy().astype(float)

    source_groups = np.concatenate([
        (source_labels == 1)[:, np.newaxis], # pattern
        (source_labels == 0)[:, np.newaxis], # pattern
    ], axis=1).copy()
    target_groups = np.concatenate([
        (target_labels == 1)[:, np.newaxis], # background
        (target_labels == 0)[:, np.newaxis], # background
    ], axis=1).copy()

    source_captions = pickle.load(open("../data/imagenetx/source_captions.pkl", "rb"))
    target_captions = pickle.load(open("../data/imagenetx/target_captions.pkl", "rb"))

    source_captions = [", ".join(cap.split(", ")[:10]) for cap in source_captions]
    target_captions = [", ".join(cap.split(", ")[:10]) for cap in target_captions]

    ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english') #, max_features=2000)
    vectorized_data = ngram_vectorizer.fit_transform(source_captions + target_captions).toarray()

    source = vectorized_data[:len(source_captions), :].astype(float)
    target = vectorized_data[len(source_captions):, :].astype(float)
    features = ngram_vectorizer.get_feature_names_out()
    # print(features)

    X = np.concatenate([source, target], axis=0)
    sel = SelectKBest(chi2, k=50).fit(X, np.concatenate([np.zeros(source.shape[0]), np.ones(target.shape[0])], axis=0))
    print(features[sel.get_support()])
    features = features[sel.get_support()]
    source = source[:298, sel.get_support()]
    target = target[:298, sel.get_support()]
    id_scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    print(source.shape[0], target.shape[0])

    return ["int"] * source.shape[1], id_scaler, source, target, source_groups, target_groups, features


def load_breast():
    COLUMN_NAMES = [
        "diagnosis", "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"
    ]
    raw_data = pd.read_csv(
        "../data/breast_cancer/data.csv",
        index_col=0)
    raw_data.drop(raw_data[(raw_data == '?').sum(axis=1) != 0].index, inplace=True)
    raw_data = raw_data[COLUMN_NAMES].dropna()

    bc_source_df = raw_data.query('diagnosis == "B"').sample(212)
    bc_target_df = raw_data.query('diagnosis == "M"').sample(212)
    bc_source = bc_source_df.drop(columns='diagnosis').to_numpy().astype(float)
    bc_target = bc_target_df.drop(columns='diagnosis').to_numpy().astype(float)

    bc_scaler = preprocessing.StandardScaler().fit(bc_source)
    # bc_source = bc_scaler.transform(bc_source)
    # bc_target = bc_scaler.transform(bc_target)

    bc_source_labels = np.concatenate([
        (bc_source[:, i] >= np.percentile(bc_source[:, i], 75)).astype(int)[:, np.newaxis] for i in range(1, bc_source.shape[1])],
        axis=1)
    bc_target_labels = np.concatenate([
        (bc_target[:, i] >= np.percentile(bc_target[:, i], 75)).astype(int)[:, np.newaxis] for i in range(1, bc_source.shape[1])],
        axis=1)
    bc_feature_names = COLUMN_NAMES[1:]
    bc_feasible_names = ["radius**2 / area above third quartile", "radius**2 / area between first and third quartile", "radius**2 / area below first quartile",]
    bc_source_feasible_groups = np.concatenate([
        (((bc_source[:, 0]**2) / bc_source[:, 3]) > np.percentile(((bc_source[:, 0]**2) / bc_source[:, 3]), 75)).astype(int)[:, np.newaxis],
        (
            (((bc_source[:, 0]**2) / bc_source[:, 3]) >= np.percentile(((bc_source[:, 0]**2) / bc_source[:, 3]), 25)) &
            (((bc_source[:, 0]**2) / bc_source[:, 3]) <= np.percentile(((bc_source[:, 0]**2) / bc_source[:, 3]), 75))
        ).astype(int)[:, np.newaxis],
        (((bc_source[:, 0]**2) / bc_source[:, 3]) < np.percentile(((bc_source[:, 0]**2) / bc_source[:, 3]), 25)).astype(int)[:, np.newaxis],
    ], axis=1)
    bc_target_feasible_groups = np.concatenate([
        (((bc_target[:, 0]**2) / bc_target[:, 3]) > np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 75)).astype(int)[:, np.newaxis],
        (
            (((bc_target[:, 0]**2) / bc_target[:, 3]) >= np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 25)) &
            (((bc_target[:, 0]**2) / bc_target[:, 3]) <= np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 75))
        ).astype(int)[:, np.newaxis],
        (((bc_target[:, 0]**2) / bc_target[:, 3]) < np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 25)).astype(int)[:, np.newaxis],
    ], axis=1)
    
    types = ["float"] * bc_source.shape[1]
    return types, bc_scaler, bc_source, bc_target, bc_source_feasible_groups, bc_target_feasible_groups, bc_feature_names

def load_adult():
    COLUMN_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income']
    raw_data = pd.read_csv("../data/adult/adult.data", names=COLUMN_NAMES, skipinitialspace=True)
    subset = ["age", "education-num", "race", "sex", "income", "marital-status", "occupation", "workclass"]
    raw_data = raw_data[subset]
    raw_data = pd.get_dummies(raw_data, columns=[
        "workclass", "marital-status", "occupation",
    ])
    binarizer = LabelBinarizer(neg_label=0, pos_label=1)
    raw_data["income"] = binarizer.fit_transform(raw_data["income"])
    raw_data["sex"] = binarizer.fit_transform(raw_data["sex"])
    raw_data["race"] = raw_data["race"].map(lambda v: 1 if v == "White" else 0)
    labels = raw_data["income"]
    raw_data = raw_data.fillna(0)
    adult_raw_data = raw_data
    adult_targets = raw_data["income"]

    adult_source_data = raw_data[raw_data["income"] == 1]
    adult_target_data = raw_data[raw_data["income"] == 0]
    samples = min(adult_source_data.shape[0], adult_target_data.shape[0])
    adult_source_data = adult_source_data.sample(samples, replace=False)
    adult_target_data = adult_target_data.sample(samples, replace=False)
    adult_source = adult_source_data.to_numpy().astype(float)[:, [i for i in range(adult_source_data.shape[1]) if i != 4]]
    adult_target = adult_target_data.to_numpy().astype(float)[:, [i for i in range(adult_source_data.shape[1]) if i != 4]]
    adult_feature_names = [adult_source_data.columns[i] for i in range(adult_source_data.shape[1]) if i != 4]
    adult_source_labels = adult_source_data.to_numpy().astype(int)[:, 4:]
    adult_target_labels = adult_target_data.to_numpy().astype(int)[:, 4:]
    adult_feasible_names = [
        "male",
        "female"
    ]
    adult_source_feasible_groups = np.concatenate([
        (adult_source[:, 3] == 1).astype(int)[:, np.newaxis],
        (adult_source[:, 3] == 0).astype(int)[:, np.newaxis],
    ], axis=1)
    adult_target_feasible_groups = np.concatenate([
        (adult_target[:, 3] == 1).astype(int)[:, np.newaxis],
        (adult_target[:, 3] == 0).astype(int)[:, np.newaxis],
    ], axis=1)
    adult_labels = np.concatenate([adult_source_labels, adult_target_labels])
    adult_scaler = MaxAbsScaler().fit(np.concatenate([adult_source, adult_target]))
    # adult_source = adult_scaler.transform(adult_source)
    # adult_target = adult_scaler.transform(adult_target)

    types = ["int", "int"] + ["binary"] * (adult_source.shape[1] - 2)

    return types, adult_scaler, adult_source, adult_target, adult_source_feasible_groups, adult_target_feasible_groups, adult_feature_names

def perturb_features(data, types, rs=[1]):
    print("rs:", rs)
    rand = np.random.uniform(size=data.shape)
    nonzero_feats = np.nonzero(np.sum(data, axis=0) > 0)[0]
    perturb_feats = np.random.choice(nonzero_feats, size=int(3 * nonzero_feats.shape[0]/4), replace=False) 
    for r in rs: #, 2, 4, 8, 16]:
        perturbation = np.zeros(data.shape)
        for perturb_feat in perturb_feats:
            if types[perturb_feat] == "binary":
                # print(perturb_feat, "binary")
                if np.random.choice([-1, 1]) == 1: #np.sum(data[:, perturb_feat]) > data.shape[0] / 2:
                    one_mask = data[:, perturb_feat] == 1
                    perturb_mask = (rand[:, perturb_feat] < (r / 100)).astype(bool)
                    perturbation[:, perturb_feat][one_mask & perturb_mask] = -1
                    # print(np.sum(one_mask & perturb_mask))
                else:
                    zero_mask = data[:, perturb_feat] == 0
                    perturb_mask = (rand[:, perturb_feat] < (r / 100)).astype(bool)
                    perturbation[:, perturb_feat][zero_mask & perturb_mask] = 1
                    # print(np.sum(zero_mask & perturb_mask))
            elif types[perturb_feat] == "int":
                # print(perturb_feat, "int")
                if np.random.choice([-1, 1]) == 1:
                    stdev = 1
                    perturb_mask = (rand[:, perturb_feat] < (r / 100)).astype(bool)
                    perturbation[:, perturb_feat][perturb_mask] = stdev
                else:
                    stdev = 1
                    perturb_mask = (rand[:, perturb_feat] < (r / 100)).astype(bool)
                    pos_mask = data[:, perturb_feat] >= stdev
                    perturbation[:, perturb_feat][pos_mask & perturb_mask] = -stdev
            else:
                stdev = 0.05 * np.std(data[:, perturb_feat])
                # print(perturb_feat, "float", stdev)
                perturb_mask = (rand[:, perturb_feat] < ((r) / 100)).astype(bool)
                pos_mask = data[:, perturb_feat] >= stdev
                perturbation[:, perturb_feat][pos_mask & perturb_mask] = stdev * np.random.choice([-1, 1])
        yield r, perturbation


def train_method(dataset, method, scaler, types, source, target, source_groups, target_groups, init_source_t=None):
    output_dict = {}
    if method == "kmeans":
        if dataset == "adult":
            lr = 10.0
        elif dataset == "breast":
            lr = 10.0
        elif dataset == "imagenet":
            lr = 150.0
        else:
            lr = 20.0

        x_s, centroids, shifts = group_kmeans_shift_transport(scaler.transform(source), scaler.transform(target), source_groups, target_groups, source.shape[1], 20, lr, 100, init_clusters=init_source_t)
        x_s = scaler.inverse_transform(x_s)
        output_dict['centroids'] = centroids
        output_dict['shift'] = shifts
    elif method == "ot":
        if dataset == "adult":
            lr = 0.05
        elif dataset == "breast":
            lr = 1.0
        elif dataset == "imagenet":
            lr = 0.5
        else:
            lr = 0.1

        x_s = group_feature_transport(scaler.transform(source), scaler.transform(target), source_groups, target_groups, source.shape[1], lr, 100, init_x_s=scaler.transform(init_source_t) if init_source_t is not None else None)
        x_s = scaler.inverse_transform(x_s)
        # shifts = x_s - source
    elif method == "dice":
        # source = scaler.inverse_transform(source)
        # target = scaler.inverse_transform(target)

        np.random.seed(42)
        source_selection = np.random.choice(np.arange(source.shape[0]), size=min(300, source.shape[0]), replace=False)

        full_data = np.concatenate([source, target], axis=0)
        full_df_dict = {}
        for idx in range(source.shape[1]):
            feature_col = full_data[:,idx]
            full_df_dict[f"feat_{idx}"] = feature_col
    
        labels = np.concatenate([np.ones(source.shape[0]), np.zeros(target.shape[0])])
        print(labels.shape, full_data.shape)
        full_df_dict["_label"] = labels
        full_df = pd.DataFrame(full_df_dict, dtype=float)
        full_df["_label"] = full_df["_label"].astype(int)
        print(full_df / full_df.max())
        source_df = full_df.iloc[0:source.shape[0]]
        target_df = full_df.iloc[source.shape[0]:]

        if dataset == "adult" or dataset == "breast":
            net = FFNetwork(source.shape[1])
        else:
            net = PTLogisticRegression(source.shape[1], 1)
    
        x_train = torch.from_numpy(np.concatenate([source, target], axis=0)).float()
        x_train = x_train / torch.max(x_train, dim=0)[0]
        x_train[torch.isnan(x_train)] = 0
        group_mat = np.concatenate([source_groups, target_groups], axis=0)
        no_group = 1 - (np.sum(group_mat, axis=1) > 0)[:, np.newaxis]
        group_mat = torch.from_numpy(np.concatenate([group_mat, no_group], axis=1)).double()
        y = torch.from_numpy(np.concatenate([np.ones(source.shape[0]), np.zeros(target.shape[0])], axis=0)).float()[:, None]

        if dataset == "adult":
            lr = 0.05
            wreg = 1e-4
        elif dataset == "breast":
            lr = 0.2
            wreg = 1e-4
        elif dataset == "imagenet":
            lr = 0.5
            wreg = 1e-3
        else:
            lr = 0.5
            wreg = 5e-5

        if source_groups.sum() == source_groups.shape[0]:
            print("Regular training")
            regular_training(
                net,
                x_train,
                y,
                group_mat,
                200,
                lr,
                wreg)
        else:
            print("DRO training")
            dro_training(
                net, 
                x_train,
                y,
                group_mat,
                200,
                lr,
                wreg)
    
        x_s = get_dice_transformed(
                net,
                full_df / full_df.max(),
                (source_df / full_df.max()).drop(columns=['_label']).iloc[source_selection],
                "_label",
                0).to_numpy()
        x_s = x_s * full_df.drop(columns=['_label']).max().to_numpy()
        # shifts = x_s - source[source_selection]
    else:
        raise ValueError("Invalid method")


    # inverse_x_s = scaler.inverse_transform(x_s)
    # inverse_shifts = scaler.inverse_transform(shifts)
    for type, i in zip(types, range(source.shape[1])):
        if type == "int":
            x_s[:, i] = np.round(x_s[:, i])
        if type == "binary":
            x_s[:, i] = np.round(x_s[:, i])
            x_s[:, i] = np.clip(x_s[:, i], 0, 1)
    # x_s_round = scaler.transform(inverse_x_s)
    if method == "dice":
        output_dict['shifts'] = x_s - source[source_selection]
    else:
        output_dict['shifts'] = x_s - source
    return output_dict


def train(dataset, method, scaler, types, orig_source, orig_target, source, target, source_groups, target_groups):
    if args.no_groups:
        X0_embeddings = extract_sentence_embeddings(list(orig_source))
        X1_embeddings = extract_sentence_embeddings(list(orig_target))
        all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
        source_groups = all_groups[:len(X0_embeddings)]
        target_groups = all_groups[len(X0_embeddings):]

    print(source.shape)
    no_group_s = np.ones((source.shape[0], 1))
    no_group_t = np.ones((target.shape[0], 1))
    if False and os.path.exists(f"../data/robustness/{method}/{dataset}_shifts.pkl"):
        ceds = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts.pkl", "rb"))
        centriods = pickle.load(open(f"../data/robustness/{method}/{dataset}_shift.pkl", "rb"))
        grceds = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts_g.pkl", "rb"))
        g_centriods = pickle.load(open(f"../data/robustness/{method}/{dataset}_shift.pkl", "rb"))
    else:
        out = train_method(dataset, method, scaler, types, source, target, no_group_s, no_group_t)
        ceds = out['shifts']
        if method == "kmeans":
            init = out["shift"]
        elif method == "ot":
            init = out["shifts"]
        else:
            init = None
        # centriods = out['centroids']
        for name, val in out.items():
            with open(f"../data/robustness/{method}/{dataset}_{name}.pkl", "wb") as f:
                pickle.dump(val, f)

        out = train_method(dataset, method, scaler, types, source, target, source_groups, target_groups)
        grceds = out['shifts']
        if method == "kmeans":
            init_g = out["shift"]
        elif method == "ot":
            init_g = out["shifts"]
        else:
            init_g = None
        # g_centriods = out['centroids']
        for name, val in out.items():
            with open(f"../data/robustness/{method}/{dataset}_{name}_g.pkl", "wb") as f:
                pickle.dump(val, f)

    for iter in range(3):
        np.random.seed(iter)
        for r, perturbation in perturb_features(source, types, rs=[1]):
            print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))

            out = train_method(dataset, method, scaler, types, source + perturbation, target, no_group_s, no_group_t) #, init_source_t=init)
            print("CEDS diff:", np.linalg.norm(out['shifts'] - ceds, ord=2) / np.linalg.norm(perturbation, ord=2))
            for name, val in out.items():
                with open(f"../data/robustness/{method}/{dataset}_{name}_perturb_{r}_{iter}.pkl", "wb") as f:
                    pickle.dump(val, f)

            out = train_method(dataset, method, scaler, types, source + perturbation, target, source_groups, target_groups) #, init_source_t=init_g)
            print("GRCEDS diff:", np.linalg.norm(out['shifts'] - grceds, ord=2) / np.linalg.norm(perturbation, ord=2))
            for name, val in out.items():
                with open(f"../data/robustness/{method}/{dataset}_{name}_g_perturb_{r}_{iter}.pkl", "wb") as f:
                    pickle.dump(val, f)

            with open(f"../data/robustness/{method}/{dataset}_perturbation_{r}_{iter}.pkl", "wb") as f:
                pickle.dump(perturbation, f)


def train_adv(dataset, method, scaler, types, orig_source, orig_target, source, target, source_groups, target_groups):
    if args.no_groups:
        X0_embeddings = extract_sentence_embeddings(list(orig_source))
        X1_embeddings = extract_sentence_embeddings(list(orig_target))
        all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
        source_groups = all_groups[:len(X0_embeddings)]
        target_groups = all_groups[len(X0_embeddings):]

    print(source.shape)

    adv_ceds = 0
    adv_grceds = 0
    source_save = source.copy()

    no_group_s = np.ones((source.shape[0], 1))
    no_group_t = np.ones((target.shape[0], 1))

    if False and os.path.exists(f"../data/robustness/{method}/{dataset}_shifts.pkl"):
        ceds = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts.pkl", "rb"))
        grceds = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts_g.pkl", "rb"))
    else:
        out = train_method(dataset, method, scaler, types, source, target, no_group_s, no_group_t)
        ceds = out['shifts']
        with open(f"../data/robustness/{method}/{dataset}_shifts.pkl", "wb") as f:
            pickle.dump(ceds, f)
        if method == "kmeans":
            init = out['shift']
        elif method == "ot":
            init = out['shifts']
        else:
            init = None

        out = train_method(dataset, method, scaler, types, source, target, source_groups, target_groups)
        grceds = out['shifts']
        with open(f"../data/robustness/{method}/{dataset}_shifts_g.pkl", "wb") as f:
            pickle.dump(grceds, f)
        if method == "kmeans":
            init_g = out['shift']
        elif method == "ot":
            init_g = out['shifts']
        else:
            init_g = None

    for i in range(3):
        np.random.seed(i)
        for iter in range(25):
            source = source_save.copy()
            for r, perturbation in perturb_features(source, types, rs=[1]):
                print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))
                print("Norm of source:", np.linalg.norm(source, ord=2))
                out = train_method(dataset, method, scaler, types, source + perturbation, target, no_group_s, no_group_t) #, init_source_t=init)
                ceds_pert = out['shifts']

                out = train_method(dataset, method, scaler, types, source + perturbation, target, source_groups, target_groups) #, init_source_t=init_g)
                grceds_pert = out['shifts']

                diff_ceds = np.linalg.norm(ceds - ceds_pert, ord=2) / np.linalg.norm(perturbation, ord=2)
                diff_grceds = np.linalg.norm(grceds - grceds_pert, ord=2) / np.linalg.norm(perturbation, ord=2)
                print("Diff in explanations:", diff_ceds, diff_grceds)
                if diff_ceds != np.nan and diff_ceds > adv_ceds:
                    adv_ceds = diff_ceds
                    with open(f"../data/robustness/{method}/{dataset}_best_ceds_perturb_{i}.pkl", "wb") as f:
                        pickle.dump(perturbation, f)
                if diff_grceds != np.nan and diff_grceds > adv_grceds:
                    adv_grceds = diff_grceds
                    with open(f"../data/robustness/{method}/{dataset}_best_grceds_perturb_{i}.pkl", "wb") as f:
                        pickle.dump(perturbation, f)
                print("Max diff:", adv_ceds, adv_grceds)
        with open(f"../data/robustness/{method}/{dataset}_adv_{i}.pkl", "wb") as f:
            pickle.dump([adv_ceds, adv_grceds], f)

            # with open(f"../data/robustness/{method}/{dataset}_perturbation_{r}_{iter}.pkl", "wb") as f:
            #     pickle.dump(perturbation, f)

def eval_adv(method, dataset):
    all_ceds, all_grceds = [], []
    for i in range(3):
        ceds, grceds = pickle.load(open(f"../data/robustness/{method}/{dataset}_adv_{i}.pkl", "rb"))
        all_ceds.append(ceds)
        all_grceds.append(grceds)
    print("Average CEDS:", np.mean(all_ceds), np.std(all_ceds))
    print("Average GR-CEDS:", np.mean(all_grceds), np.std(all_grceds))

def qualitative_eval(dataset, orig_source, orig_target, source, target, source_groups, target_groups, feature_names):
    if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_groups.pkl"):
        X0_embeddings = extract_sentence_embeddings(list(orig_source))
        X1_embeddings = extract_sentence_embeddings(list(orig_target))
        all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
        source_groups = all_groups[:len(X0_embeddings)]
        target_groups = all_groups[len(X0_embeddings):]
        with open(f"../data/robustness/kmeans/{dataset}_groups.pkl", "wb") as f:
            pickle.dump([source_groups, target_groups], f)
    else:
        source_groups, target_groups = pickle.load(open(f"../data/robustness/kmeans/{dataset}_groups.pkl", "rb"))

    no_group_s = np.ones((source.shape[0], 1))
    no_group_t = np.ones((target.shape[0], 1))

    # perturbation = pickle.load(open(f"../data/robustness/kmeans/{dataset}_best_ceds_perturb.pkl", "rb"))
    perturbation = [p for p in perturb_features(source, ["int"] * source.shape[1], rs=[5])][0][1]
    print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))
    print(perturbation.shape)
    if False and os.path.exists(f"../data/robustness/kmeans/{dataset}_shifts.pkl"):
        ceds = pickle.load(open(f"../data/robustness/kmeans/{dataset}_shifts.pkl", "rb")).round(0)
        grceds = pickle.load(open(f"../data/robustness/kmeans/{dataset}_shifts_g.pkl", "rb")).round(0)
    else:
        out = train_method(dataset, "kmeans", scaler, types, source, target, no_group_s, no_group_t)
        ceds = out['shifts']
        with open(f"../data/robustness/kmeans/{dataset}_shifts.pkl", "wb") as f:
            pickle.dump(ceds, f)
        # out = train_method(dataset, "kmeans", scaler, types, source, target, source_groups, target_groups)
        # grceds = out['shifts']
        # with open(f"../data/robustness/kmeans/{dataset}_shifts_g.pkl", "wb") as f:
        #     pickle.dump(grceds, f)
        grceds = ceds
    print("modified source shape:", ceds.shape)

    if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_best_ceds_shifts.pkl"):
        out = train_method(dataset, "kmeans", scaler, types, source + perturbation, target, no_group_s, no_group_t)
        ceds_pert = out['shifts'].round(0) + source + perturbation
        ceds_shifts = out['shifts'].round(0)
        # ceds_centroids = out['centroids']
        # with open(f"../data/robustness/kmeans/{dataset}_best_ceds_centroids.pkl", "wb") as f:
        #     pickle.dump(ceds_centroids, f)
        with open(f"../data/robustness/kmeans/{dataset}_best_ceds_shifts.pkl", "wb") as f:
            pickle.dump(ceds_shifts, f)
    else:
        # ceds_centroids = pickle.load(open(f"../data/robustness/kmeans/{dataset}_best_ceds_centroids.pkl", "rb"))
        ceds_shifts = pickle.load(open(f"../data/robustness/kmeans/{dataset}_best_ceds_shifts.pkl", "rb")).round(0)
        ceds_pert = ceds_shifts + source + perturbation
        # ceds_pert = transform_samples_kmeans(scaler.transform(source + perturbation), ceds_centroids, ceds_shift)

    if not os.path.exists(f"../data/robustness/kmeans/{dataset}_best_ceds_shifts_g.pkl"):
        out = train_method(dataset, "kmeans", scaler, types, source + perturbation, target, source_groups, target_groups) #, init_source_t=centroids_g)
        grceds_pert = out['shifts'] + source + perturbation
        grceds_shifts = out['shifts']
        # grceds_centroids = out['centroids']
        # with open(f"../data/robustness/kmeans/{dataset}_best_ceds_centroids_g.pkl", "wb") as f:
        #     pickle.dump(grceds_centroids, f)
        with open(f"../data/robustness/kmeans/{dataset}_best_ceds_shifts_g.pkl", "wb") as f:
            pickle.dump(grceds_shifts, f)
    else:
        grceds_pert = ceds_pert
        grceds_shifts = ceds_shifts
        # grceds_centroids = pickle.load(open(f"../data/robustness/kmeans/{dataset}_best_ceds_centroids_g.pkl", "rb"))
        # grceds_shifts = pickle.load(open(f"../data/robustness/kmeans/{dataset}_best_ceds_shifts_g.pkl", "rb"))
        # grceds_pert = grceds_shifts + source + perturbation
        # grceds_pert = transform_samples_kmeans(scaler.transform(source + perturbation), grceds_centroids, grceds_shift)

    # print(np.linalg.norm(ceds - ceds_pert, ord=2) / np.linalg.norm(perturbation, ord=2))
    # print(np.linalg.norm(grceds - grceds_pert, ord=2) / np.linalg.norm(perturbation, ord=2))

    def is_adult_feasible_change(orig, new):
        orig_sex = (orig[:, 3].round(0))[:, np.newaxis]
        new_sex = (new[:, 3].round(0))[:, np.newaxis]

        return (
            np.all(new_sex == orig_sex, axis=1) )#&

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

    def is_nlp_feasible_change(orig_groups, new_groups):
        feas = np.sum(orig_groups == new_groups, axis=1) == orig_groups.shape[1]
        return feas, new_groups

    if dataset == "nlp":
        if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_new_groups.pkl"):
            neigh = KNeighborsClassifier(n_neighbors=1).fit(X1_embeddings, np.arange(len(X1_embeddings)))
            cf_source_trans = [edit_text(s, feature_names, diff) for s, diff in zip(orig_source, ceds)]
            cf_source_trans_emb = extract_sentence_embeddings(cf_source_trans)
            new_groups = target_groups[neigh.predict(cf_source_trans_emb)]
            with open(f"../data/robustness/kmeans/{dataset}_new_groups.pkl", "wb") as f:
                pickle.dump(new_groups, f)
        else:
            new_groups = pickle.load(open(f"../data/robustness/kmeans/{dataset}_new_groups.pkl", "rb"))

        feasible, new_groups = is_nlp_feasible_change(source_groups, new_groups)
        print("Original Percent feasible shift:", (np.sum(feasible) / feasible.shape[0]) * 100)

        if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_new_groups_pert.pkl"):
            neigh = KNeighborsClassifier(n_neighbors=1).fit(X1_embeddings, np.arange(len(X1_embeddings)))
            cf_source_trans = [edit_text(s, feature_names, diff) for s, diff in zip(orig_source, perturbation + ceds_shifts)]
            cf_source_trans_emb = extract_sentence_embeddings(cf_source_trans)
            new_groups_pert = target_groups[neigh.predict(cf_source_trans_emb)]
            with open(f"../data/robustness/kmeans/{dataset}_new_groups_pert.pkl", "wb") as f:
                pickle.dump(new_groups_pert, f)
        else:
            new_groups_pert = pickle.load(open(f"../data/robustness/kmeans/{dataset}_new_groups_pert.pkl", "rb"))

        if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_perturb_groups.pkl"):
            neigh_source = KNeighborsClassifier(n_neighbors=1).fit(X0_embeddings, np.arange(len(X0_embeddings)))
            cf_source_perturb = [edit_text(s, feature_names, diff) for s, diff in zip(orig_source, perturbation)]
            cf_source_perturb_emb = extract_sentence_embeddings(cf_source_perturb)
            perturb_groups = source_groups[neigh_source.predict(cf_source_perturb_emb)]
        else:
            perturb_groups = pickle.load(open(f"../data/robustness/kmeans/{dataset}_perturb_groups.pkl", "rb"))

        feasible_pert, new_groups_pert = is_nlp_feasible_change(perturb_groups, new_groups_pert)
        print("Pert Percent feasible shift:", (np.sum(feasible_pert) / feasible_pert.shape[0]) * 100)
    elif dataset == "adult":
        feasible = is_adult_feasible_change(source, ceds + source)
        print("Original Percent feasible shift:", (np.sum(feasible) / feasible.shape[0]) * 100)
        feasible_pert = is_adult_feasible_change(source + perturbation, ceds_pert)
        print("Pert Percent feasible shift:", (np.sum(feasible_pert) / feasible_pert.shape[0]) * 100)

    becomes_infeasible = np.logical_and(feasible, ~feasible_pert)


    total, worst = group_percent_explained(
        source + perturbation,
        ceds_pert,
        target,
        source_groups,
        target_groups,
        [f"f{i}" for i in range(source_groups.shape[1])])
    print(total, worst)
    total, worst = group_percent_explained(
        source + perturbation,
        grceds_pert,
        target,
        source_groups,
        target_groups,
        [f"f{i}" for i in range(source_groups.shape[1])])
    print(total, worst)

    # Print one sample for each group
    for i in range(source_groups.shape[1]):
        print(f"Group {i}")
        print(orig_source[source_groups[:, i] == 1][0])
        print()

    # diffs = np.abs(ceds_shift - grceds_shift)
    # print(np.unique(ceds, axis=0))
    diffs = np.abs(ceds - ceds_shifts)
    sort_idx = np.argsort(-diffs, axis=1)
    sort_idx0 = np.argsort(-np.sum(np.abs(perturbation[becomes_infeasible]), axis=1)).flatten()
    features = feature_names
    j = 0
    for group, new_group, new_group_pert, feas, s_text, t_text, sample, sample_perturb, shift, shift_orig in np.array(list(zip(
        source_groups[becomes_infeasible], new_groups[becomes_infeasible], new_groups_pert[becomes_infeasible],
        feasible_pert[becomes_infeasible], orig_source[becomes_infeasible], orig_target[becomes_infeasible], source[becomes_infeasible],
        (perturbation)[becomes_infeasible], ceds_shifts[becomes_infeasible], ceds[becomes_infeasible])))[sort_idx0]:

        # if np.sum(np.abs(sample_perturb)) > 1:
        #     continue

        # print([f"{features[i]}: {centroid[i].round(1)}" for i in np.nonzero(centroid.round(1))[0]])
        print(f"Group {np.nonzero(group)[0]} -> {np.nonzero(new_group)[0]} -> {np.nonzero(new_group_pert)[0]}")
        print("Feasible:", feas)
        print("Source:", s_text)
        print("Source features:", [f"{features[i]}: {sample[i].round(1)}" for i in np.nonzero(sample.round(1))[0]])
        print("Orig transformed", [f"{features[i]}: {shift_orig[i].round(1)}" for i in np.nonzero(shift_orig.round(1))[0]])
        print("Pert transformed", [f"{features[i]}: {shift[i].round(1)}" for i in np.nonzero(shift.round(1))[0]])
        print("Perturbed features:", [f"{features[i]}: {sample_perturb[i].round(1)}" for i in np.nonzero(sample_perturb.round(1))[0]])
        # print(f"Sample {j}")
        # for i in sort_idx[j][:20]:
            # print(f"Shift in {features[i]} by {shift[i].round(1)}. (initial: {shift_orig[i].round(1)})")
            # print(f"Group Shift in {features[i]} by {g_shift[i].round(1)}. (initial: {g_shift_orig[i].round(1)})")
        # for i in np.argsort(-np.abs(shift_orig))[:5]:
            # print(f"Orig Shift in {features[i]} by {shift[i].round(1)}. (initial: {shift_orig[i].round(1)})")
        # j += 1
        print()

    print("----- Feasible -----")
    becomes_infeasible = np.argsort(-np.sum(abs(perturbation), axis=1)).flatten()[np.logical_and(feasible, feasible_pert)]
    for groups, feas, s_text, t_text, sample, sample_perturb, shift, shift_orig, g_shift, g_shift_orig in np.array(list(zip(
        source_groups[becomes_infeasible], feasible_pert[becomes_infeasible], orig_source[becomes_infeasible], orig_target[becomes_infeasible], source[becomes_infeasible],
        (source + perturbation)[becomes_infeasible], ceds_shifts[becomes_infeasible], ceds[becomes_infeasible], grceds_shifts[becomes_infeasible], grceds[becomes_infeasible])))[sort_idx0][:5]:
        print("Groups:", groups)
        print("Feasible:", feas)
        print("Source:", s_text)
        print("Source features:", [f"{features[i]}: {sample[i].round(1)}" for i in np.nonzero(sample.round(1))[0]])
        print("Orig transformed", [f"{features[i]}: {shift_orig[i].round(1)}" for i in np.nonzero(shift_orig.round(1))[0]])
        print("Pert transformed", [f"{features[i]}: {shift[i].round(1)}" for i in np.nonzero(shift.round(1))[0]])
        print("Perturbed features:", [f"{features[i]}: {sample_perturb[i].round(1)}" for i in np.nonzero(sample_perturb.round(1))[0]])



    # print()
    # diffs = np.abs(grceds - grceds_shift)
    # sort_idx = np.argsort(-diffs, axis=1)
    # j = 0
    # for centroid, shift, shift_orig in zip(grceds_centroids[:5], grceds_shift[:5], grceds[:5]):
    #     print([f"{features[i]}: {centroid[i].round(1)}" for i in np.nonzero(centroid.round(1))[0]])
    #     for i in sort_idx[j][:5]:
    #         print(f"Shift in {features[i]} by {shift[i].round(1)}. (initial: {shift_orig[i].round(1)} {diffs[j][i]})")
    #     j += 1



def eval(method, dataset, orig_source, orig_target, source_groups, selection):
    if args.no_groups:
        X0_embeddings = extract_sentence_embeddings(list(orig_source))
        X1_embeddings = extract_sentence_embeddings(list(orig_target))
        all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
        source_groups = all_groups[:len(X0_embeddings)]
        target_groups = all_groups[len(X0_embeddings):]

    all_base_deltas = []
    base_std = []
    all_group_deltas = []
    group_std = []

    # if method == "kmeans":
    #     centroids = pickle.load(open(f"../data/robustness/{method}/{dataset}_centroids.pkl", "rb"))
    #     centroids_g = pickle.load(open(f"../data/robustness/{method}/{dataset}_centroids_g.pkl", "rb"))
    base_exp = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts.pkl", "rb"))
    group_exp = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts_g.pkl", "rb"))
    # if method == "kmeans":
    #     source_transformed = transform_samples_kmeans(source, centroids, shifts)
    #     source_transformed_g = transform_samples_kmeans(source, centroids_g, shifts_g)
    # else:
    #     source_transformed = source + shifts
    #     source_transformed_g = source + shifts_g

    for r in [1]:
        base_deltas = []
        group_deltas = []
        for iter in range(3):
            perturbation = pickle.load(open(f"../data/robustness/{method}/{dataset}_perturbation_{r}_{iter}.pkl", "rb")) #[selection]
            # if method == "dice":
            #     perturbation = perturbation[selection]
            print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))

            # print("Total percent changed:", np.sum((np.sum(perturbation, axis=1) > 0)) / perturbation.shape[0])
            # for gid in range(source_groups.shape[1]):
            #     print(f"Percent changed in group {gid}:",
            #           np.sum((np.sum(perturbation[source_groups[:, gid] == 1], axis=1) > 0)) / np.sum(source_groups[:, gid] == 1))
            # if method == "kmeans":
            #     centroids_perturb = pickle.load(open(f"../data/robustness/{method}/{dataset}_centroids_perturb_{r}_{iter}.pkl", "rb"))
            #     centroids_g_perturb = pickle.load(open(f"../data/robustness/{method}/{dataset}_centroids_g_perturb_{r}_{iter}.pkl", "rb"))

            base_perturb_exp = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts_perturb_{r}_{iter}.pkl", "rb"))
            group_perturb_exp = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts_g_perturb_{r}_{iter}.pkl", "rb"))

            # print("shifts", shifts.shape)
            # if method == "kmeans":
            #     perturb_transformed = transform_samples_kmeans(source + perturbation, centroids_perturb, shifts_perturb)
            #     perturb_transformed_g = transform_samples_kmeans(source + perturbation, centroids_g_perturb, shifts_g_perturb)
            # else:
            #     print(source.shape, perturbation.shape)
            #     perturb_transformed = source + perturbation + shifts_perturb
            #     perturb_transformed_g = source + perturbation + shifts_g_perturb

            # base_exp = source - source_transformed
            # base_perturb_exp = source + perturbation - perturb_transformed

            # group_exp = source - source_transformed_g
            # group_perturb_exp = source + perturbation - perturb_transformed_g

            base_delta = np.linalg.norm(base_exp - base_perturb_exp, ord=2) / np.linalg.norm(perturbation, ord=2)
            group_delta = np.linalg.norm(group_exp - group_perturb_exp, ord=2) / np.linalg.norm(perturbation, ord=2)
            print(base_delta, group_delta)
            base_deltas.append(base_delta)
            group_deltas.append(group_delta)

        all_base_deltas.append(np.mean(base_deltas))
        base_std.append(np.std(base_deltas))
        all_group_deltas.append(np.mean(group_deltas))
        group_std.append(np.std(group_deltas))
        print("Base delta:", np.mean(base_deltas), np.std(base_deltas))
        print("Group delta:", np.mean(group_deltas), np.std(group_deltas))
        print()
    print(len(all_base_deltas), len(all_group_deltas))
    out_dict = [{"r": r, "base_delta": base_delta, "base_std": base_std, "group_delta": group_delta, "group_std": group_std} for r, base_delta, base_std, group_delta, group_std in zip([1, 2, 4, 8, 16], all_base_deltas, base_std, all_group_deltas, group_std)]
    with open(f"../data/robustness/{dataset}_{method}_out.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["r", "base_delta", "base_std", "group_delta", "group_std"])
        writer.writeheader()
        for row in out_dict:
            writer.writerow(row)

    print(all_base_deltas)
    print(all_group_deltas)
    # plt.style.use('seaborn-whitegrid')
    # plt.plot([1, 2, 4, 8, 16], all_base_deltas, label=f"{method}")
    # plt.plot([1, 2, 4, 8, 16], all_group_deltas, label=f"WG-{method}")
    # plt.fill_between([1, 2, 4, 8, 16], np.array(all_base_deltas) - np.array(base_std), np.array(all_base_deltas) + np.array(base_std), alpha=0.2)
    # plt.fill_between([1, 2, 4, 8, 16], np.array(all_group_deltas) - np.array(group_std), np.array(all_group_deltas) + np.array(group_std), alpha=0.2)
    # plt.xlabel("r")
    # plt.ylabel("L1 distance")
    # plt.legend()
    # tikzplotlib.save(f"../figures/{dataset}_{method}_perturb.tex")

def get_perturbation(source):
    perturbation = np.random.normal(0, 1, source.shape)
    perturbation /= np.linalg.norm(perturbation, axis=1, keepdims=True, ord=1) / 2
    return perturbation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # a flag argument
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--method')
    parser.add_argument('--dataset')
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--qual', action='store_true')
    parser.add_argument('--no_groups', action='store_true')
    args = parser.parse_args()

    if args.dataset == "adult":
        types, scaler, source, target, source_groups, target_groups, feature_names = load_adult()
    elif args.dataset == "breast":
        types, scaler, source, target, source_groups, target_groups, feature_names = load_breast()
    elif args.dataset == "imagenet":
        types, scaler, source, target, source_groups, target_groups, feature_names = load_imagenet()
    elif args.dataset == "nlp":
        types, scaler, orig_source, orig_target, source, target, source_groups, target_groups, feature_names = load_nlp()
    else:
        raise NotImplementedError

    if args.method != "dice":
        sample_size = 1000
        np.random.seed(0)
        subset = np.random.permutation(source.shape[0])[:sample_size]
    else:
        subset = np.arange(source.shape[0])

    if args.train:
        Path(f"../data/robustness/{args.method}").mkdir(parents=True, exist_ok=True)

        if args.adv and args.dataset != "nlp":
            train_adv(args.dataset, args.method, scaler, types, None, None, source[subset], target[subset], source_groups[subset], target_groups[subset])
        elif args.adv:
            train_adv(args.dataset, args.method, scaler, types, orig_source[subset], orig_target[subset], source[subset], target[subset], source_groups[subset], target_groups[subset])
        elif args.dataset != "nlp":
            train(args.dataset, args.method, scaler, types, None, None, source[subset], target[subset], source_groups[subset], target_groups[subset])
        else:
            train(args.dataset, args.method, scaler, types, orig_source[subset], orig_target[subset], source[subset], target[subset], source_groups[subset], target_groups[subset])
    elif args.qual:
        qualitative_eval(args.dataset, orig_source[subset], orig_target[subset], source[subset], target[subset], source_groups[subset], target_groups[subset], feature_names)
    else:
        if args.method == "dice":
            np.random.seed(42)
            subsubset = np.random.choice(np.arange(subset.shape[0]), size=min(300, subset.shape[0]), replace=False)
            subset = subset[subsubset]
        else:
            subsubset = subset
        # else:
        #     sample_size = -1
        #     np.random.seed(0)
        #     subset = np.random.permutation(source.shape[0])[:sample_size]

        if args.adv:
            eval_adv(args.method, args.dataset)
        else:
            print(subset.shape)
            if args.dataset != "nlp":
                eval(args.method, args.dataset, None, None, source_groups[subset], subsubset)
            else:
                eval(args.method, args.dataset, orig_source[subset], orig_target[subset], source_groups[subset], subsubset)
