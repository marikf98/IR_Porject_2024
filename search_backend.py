# Import all libraries needed
import numpy as np
import math
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *
from nltk.corpus import stopwords
from inverted_index_gcp import *
from pathlib import Path
import requests
from collections import Counter
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed


bucket_name = 'h3_2024'


def get_pagerank():
    # Read the doc_id, pagerank_score csv from bucket
    pagerank_df = pd.read_csv('gs://h3_2024/pr/part-00000-c2d78cc0-c046-4f5b-ae54-6ed6d5996a0f-c000.csv.gz',
                              header=None)
    # Set columns and index
    pagerank_df.columns = ["DocumentID", "PageRank"]
    pagerank_df.set_index('DocumentID', inplace=True)
    # Normalize the pagerank scores.
    min_rank = pagerank_df['PageRank'].min()
    max_rank = pagerank_df['PageRank'].max()
    pagerank_df['PageRank'] = (pagerank_df['PageRank'] - min_rank) / (max_rank - min_rank)
    return pagerank_df


def read_pickle_from_gcp(file_name):
    download_path = file_name + ".pickle"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(download_path)
    raw_data = blob.download_as_bytes()
    data = pickle.loads(raw_data)
    return data


# Import indexes
text_index = InvertedIndex.read_index(".", "text", bucket_name)
title_index = InvertedIndex.read_index(".", "title", bucket_name)
# Get the pagerank scores from the bucket
pagerank_df = get_pagerank()
# Import doc_id, title dict
doc_id_title_dict = read_pickle_from_gcp("doc_id_title_dict")


# Tokenize and Stemming

# Unite english stopwords and corpus stopwords into one list
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower())]


# Use porter stemmer on the tokens
stemmer = PorterStemmer()


def filter_tokens(tokens, tokens2remove=None, use_stemming=False):
    """ The function takes a list of tokens, filters out `tokens2remove` and
      stem the tokens using `stemmer`.
    Parameters:
    -----------
    tokens: list of str.
    Input tokens.
    tokens2remove: frozenset.
    Tokens to remove (before stemming).
    use_stemming: bool.
    If true, apply stemmer.stem on tokens.
    Returns:
    --------
    list of tokens from the text.
    """
    if tokens2remove:
        tokens = [token for token in tokens if token not in tokens2remove]
    if use_stemming:
        tokens = map(stemmer.stem, tokens)
    return tokens


def calculate_bm(inverted, postings_list):
    average_document_length = inverted.avg_doc_len
    doc_len_dict = inverted.doc_len_dict
    # Parameters for the BM25
    k = 1
    b = 0.75

    # turning each element of the posting_list from [(doc_id, tf), (doc_id, tf), ...]
    # to [(doc_id, bm25), (doc_id, bm25), ...]
    # tf -> (doc_id_tf_pair[1]/doc_len_dict[doc_id_tf_pair[0]])
    # B -> ((1-b+b*(doc_len_dict[doc_id_tf_pair[0]]))/average_document_length)
    # df -> len(post_list)

    df = len(postings_list)
    N = len(doc_len_dict)
    postings_list_bm = [(doc_id_tf_pair[0],
                            ((k + 1) * doc_id_tf_pair[1]) / ((1 - b + b * (doc_len_dict[doc_id_tf_pair[0]]/average_document_length)) * k + doc_id_tf_pair[1])
                            * math.log((N + 1) / df)
                            )for doc_id_tf_pair in postings_list]
    return postings_list_bm


def calc_tf_query(list_of_tokens):
    k3 = 0.5
    list_of_tokens = list(list_of_tokens)
    return {token: ((list_of_tokens.count(token) / len(list_of_tokens)) * (k3 + 1)) /
                   ((list_of_tokens.count(token) / len(list_of_tokens)) + k3)
            for token in list_of_tokens}


def search_text(final_query, index_text, base_dir, bucket_name):
    docs_rank = {}
    sum_of_bm25ranks = 0
    term_posting_list_body_bm = []
    for term, query_term_tf in final_query.items():
        # Searching in body
        try:
            term_posting_list_body = sorted(index_text.read_a_posting_list(base_dir, term, bucket_name), key=lambda x: x[1],reverse=True)[:100000]
        except:
            continue
        term_posting_list_body_bm = calculate_bm(index_text, term_posting_list_body)
        sum_of_bm25ranks = 0
        # Ranking for body
        for doc_id_bm25_pair in term_posting_list_body_bm:
            if doc_id_bm25_pair[0] in docs_rank:
                docs_rank[doc_id_bm25_pair[0]] += doc_id_bm25_pair[1] * query_term_tf
            else:
                try:
                    rank = pagerank_df.loc[doc_id_bm25_pair[0], "PageRank"]
                except:
                    rank = 0
                docs_rank[doc_id_bm25_pair[0]] = doc_id_bm25_pair[1] * query_term_tf # + rank * pagerank_weight
            sum_of_bm25ranks += doc_id_bm25_pair[1] * query_term_tf

    avg_of_bm25ranks = sum_of_bm25ranks / len(term_posting_list_body_bm)
    docs_rank[-999] = avg_of_bm25ranks
    return docs_rank


def search_title(final_query, index_title, base_dir, bucket_name):
    title_rank = {}
    for term, query_term_tf in final_query.items():
        # Searching in title
        try:
            term_posting_list_title = index_title.read_a_posting_list(base_dir, term, bucket_name)
        except:
            continue
        # Ranking for title
        for doc_id_tf_pair in term_posting_list_title:
            if doc_id_tf_pair[0] in title_rank:
                title_rank[doc_id_tf_pair[0]] += 1
            else:
                title_rank[doc_id_tf_pair[0]] = 1
    return title_rank


def combine_scores(title_scores, body_scores):
    title_weight, text_weight = 0.2, 0.6
    avg_of_bm25ranks = body_scores.get(-999, 1)
    final_scores = {}
    for doc_id in set(title_scores) | set(body_scores):
        final_scores[doc_id] = (title_scores.get(doc_id, 0) * title_weight * avg_of_bm25ranks) + (body_scores.get(doc_id, 0) * text_weight)
    return final_scores


def search_backend(query, index_title, index_text, pagerank_df, base_dir, bucket_name):
    # tokenize the query from string to list of tokens.
    query_tokens = tokenize(query)
    pagerank_weight = 0.2
    # filtering stopwords and stemming
    query_filtered = filter_tokens(query_tokens, all_stopwords, True)
    # calculate the tf for the query. result is dict of {term : tf}
    final_query = calc_tf_query(query_filtered)
    tasks = {
        'title': (search_title, (final_query, index_title, base_dir, bucket_name)),
        'text': (search_title, (final_query, index_text, base_dir, bucket_name)),
    }
    results = {
        'title': {},
        'body': {},
    }
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Future-to-key mapping
        future_to_key = {
            executor.submit(task[0], *task[1]): key for key, task in tasks.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                print(f'{key} generated an exception: {exc}')

    final_scores = combine_scores(results['title'], results['body'])
    for doc_id in final_scores:
        try:
            rank = pagerank_df.loc[doc_id, "PageRank"]
        except:
            rank = 0
        final_scores[doc_id] += rank * pagerank_weight
    docs_rank = sorted(final_scores.items(), key=lambda doc_grade: doc_grade[1], reverse=True)[:100]
    return [(str(doc_id_rank_pair[0]), doc_id_title_dict[doc_id_rank_pair[0]]) for doc_id_rank_pair in docs_rank]


def search_in_corpus(query):
    return search_backend(query, title_index, text_index, pagerank_df, ".", bucket_name)
