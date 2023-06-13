from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import dice_ml
import pandas as pd


def get_dice_transformed(model, full_data, source_data, target_name, desired_class):
    d = dice_ml.Data(dataframe=full_data,
                     continuous_features=list(full_data.drop(columns=[target_name]).columns),
                     outcome_name=target_name)
    m = dice_ml.Model(model=model, backend="PYT")
    exp = dice_ml.Dice(d, m, method="gradient")
    feature_weights = {col_name: 10 for col_name in full_data.drop(columns=[target_name]).columns}
    print("new")
    e1 = exp.generate_counterfactuals(source_data,# / full_data.drop(columns=[target_name]).max(),
                                      total_CFs=1,
                                      min_iter=1,
                                      max_iter=100,
                                      verbose=False,
                                      desired_class=desired_class,
                                      feature_weights=feature_weights,
                                      posthoc_sparsity_algorithm="binary",
                                      proximity_weight=2.0,
                                      diversity_weight=1.0)
    transformed = pd.concat([e.final_cfs_df for e in e1.cf_examples_list]).drop(columns=[target_name])
    return transformed

def get_closest_target(transformed_source, full_target):
    neigh = KNeighborsClassifier(n_neighbors=1).fit(full_target, np.arange(full_target.shape[0]))
    mapping = neigh.predict(transformed_source)
    return mapping
