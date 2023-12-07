import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from embedded_topic_model.utils import preprocessing
import json
from embedded_topic_model.utils import embedding
from embedded_topic_model.models.etm import ETM
import re

def preprocess_datasets(documents):
    processed_doc = []
    for doc in documents:
        doc = doc.lower()
        doc = re.sub(r'[^A-Za-z0-9 ]+', '', doc)
        processed_doc.append(doc)
    return processed_doc

def obtain_lda_feat(documents, num_topics=20, lr=0.001, epochs=200):
    documents = preprocess_datasets(documents)
    embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(documents)
    vocabulary, train_dataset, test_dataset, train_ids, test_ids = preprocessing.create_etm_datasets(
        documents, 
        min_df=0.0, 
        max_df=1.0, 
        train_size=1, 
    )

    # Training an ETM instance
    etm_instance = ETM(
        vocabulary,
        embeddings=embeddings_mapping, # You can pass here the path to a word2vec file or
                                    # a KeyedVectors instance
        lr=lr,
        num_topics=num_topics,
        epochs=epochs,
        debug_mode=True,
        train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
                                # topic embeddings. By default, is False. If 'embeddings' argument
                                # is being passed, this argument must not be True
    )
    
    etm_instance.fit(train_dataset)
    
    return etm_instance.get_document_topic_dist(), ["topic_" + str(i) for i in range(num_topics)], train_ids, etm_instance
    
    