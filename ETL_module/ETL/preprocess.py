import json
import gc

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle



# Датафрейм действий пользователя
# def get_actions_df(path='data/sessions.json') -> pd.DataFrame:
def get_actions_df(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        sessions = json.load(f)
    session_ids = []
    items = []
    action_types = []
    for i in sessions:
        session = sessions[str(i)]['session']
        for item_id, action_type in session.items():
            session_ids.append(i)
            items.append(item_id)
            action_types.append(action_type)
        for item_id in sessions[str(i)]['order']:
            session_ids.append(i)
            items.append(item_id)
            action_types.append('order')
            
    df = pd.DataFrame(
        {
            'session_id': session_ids,
            'item_id': items,
            'action_type': action_types
        }
    )
    return df


def get_mapping(
    df: pd.DataFrame,
) -> pd.DataFrame:
    uniq_items = df['item_id'].unique()
    item_mapping = pd.DataFrame({'old': uniq_items, 'new': np.arange(uniq_items.shape[0])})
    df['item_id'] = df['item_id'].map(item_mapping.set_index('old').new)
    df['session_id'] = df['session_id'].astype('uint32')
    item_mapping['old'] = item_mapping['old'].astype('str')
    item_mapping['new'] = item_mapping['new'].astype('int')
    item_mapping.index.name = 'index'
    return df, item_mapping


def get_sparse_matrix(
    type_of_action: str,
    session_id_max: int,
    item_id_max: int,
    this_df: pd.DataFrame
) -> sp.csr_matrix:
        shape = (session_id_max + 1, item_id_max + 1)
        # из df возьмем только view
        condition = (this_df['action_type'] == type_of_action)
        session_matrix = sp.csr_matrix(
            (np.ones(np.sum(condition)),
                (
                    this_df.loc[condition, 'session_id'],
                    this_df.loc[condition, 'item_id']
                )
            ),
            shape=shape
        )
        return session_matrix


def main(
    input_data: str,
    test_df_dir: str,
    train_df_dir: str,
    sparse_matrices_dir: str,
    data_dir: str,
    w2vec_data_dir: str,
    **kwargs,
):
    # path = kwargs["path"]
    df = get_actions_df(input_data)
    df, item_mapping = get_mapping(df)
    all_sessions = np.unique(df.session_id)
    train_size = int(0.9 * all_sessions.size)
    np.random.shuffle(all_sessions)
    train_sessions = all_sessions[:train_size]
    train_df = df[df['session_id'].isin(train_sessions)]
    test_df = df[~df['session_id'].isin(train_sessions)]

    print("TEST DF SHAPE:", test_df.shape)
    print("TRAIN DF SHAPE:", train_df.shape)
    
    test_df = (
        test_df.groupby(['session_id', 'action_type'])['item_id']
        .agg(list)
        .unstack()
    )
    test_df.columns.name = None
    print(test_df.columns)
    for col in ["to_cart", "view", "order"]:
        test_df[col] = test_df[col].fillna({i: [] for i in test_df.index})

    test_df.to_parquet(test_df_dir)
    
    # Матрица взаимодействий. Отдельно для каждого типа действий
    session_id_max, item_id_max = df['session_id'].max(), df['item_id'].max()

    for action_type in ["view", "to_cart", "order"]:
        matrix = get_sparse_matrix(
            "view", session_id_max, item_id_max, train_df
        )
        similarities_matrix = cosine_similarity(
            matrix.transpose(), dense_output=False,
        )
        similarities_matrix.setdiag(0)
        sp.save_npz(f"{sparse_matrices_dir}/{action_type}_matrix.npz", matrix)
        sp.save_npz(
            f"{sparse_matrices_dir}/similarities_{action_type}.npz",
            similarities_matrix
        )
    del matrix
    del similarities_matrix
    gc.collect()

    df = train_df
    # Число действий по каждому item_id в рамках конкретного типа действий
    for action_type in ['view', 'order', 'to_cart']:
        un = df[df.action_type == action_type]['item_id'].values
        cntr = Counter(un)
        with open(f'{data_dir}/num_of_'+action_type+'.pkl', 'wb') as handle:
            pickle.dump(cntr, handle)

    #50 самых заказываемых товаров
    vals = df[df.action_type=='order']['item_id'].values
    cntr = Counter(vals)
    top50 = [i[0] for i in cntr.most_common(60)][10:]
    with open(f'{data_dir}/top50.pkl', 'wb') as handle:
        pickle.dump(top50, handle)


    # Обучающий набор данных для W2VEC.
    processed_df = pd.DataFrame()
    for action_type in ['view', 'order', 'to_cart']:
        action_column = (
            df[df['action_type'] == action_type]
            .groupby('session_id')['item_id'].agg(list)
        )
        processed_df[action_type] = action_column
        if action_type in ["view", "order", "to_cart"]:
            (
                action_column
                .map(lambda x: [str(i) for i in x])
                .to_pickle(f"{w2vec_data_dir}/{action_type}_df.pkl", protocol=3)
            )
        processed_df[action_type] = processed_df[action_type].fillna(
            {i: [] for i in processed_df.index}
        )
    item_mapping = dict(
        zip(item_mapping['new'].values, item_mapping['old'].values)
    )
    reverse_mapping = dict(
        zip(item_mapping.values(), item_mapping.keys())
    )
    
    processed_df.to_parquet(train_df_dir)
    del processed_df
    with open(f"{data_dir}/item_mapping.pkl", "wb") as handle:
        pickle.dump(item_mapping, handle)
    with open(f"{data_dir}/reverse_mapping.pkl", "wb") as handle:
        pickle.dump(reverse_mapping, handle)
    
    session_order_mat = sp.load_npz(f"{sparse_matrices_dir}/order_matrix.npz")
    for action_type in ["to_cart", "view"]:
        session_mat = sp.load_npz(
            f"{sparse_matrices_dir}/{action_type}_matrix.npz"
        )
        PAB_action = np.dot(session_mat.T, session_order_mat)
        action_norm_mat = np.sum(session_mat, axis=0)
        sp.save_npz(f"{sparse_matrices_dir}/PAB_{action_type}.npz", PAB_action)
        with open(f"{sparse_matrices_dir}/PB_{action_type}.npy", "wb") as f:
            np.save(f, action_norm_mat)
