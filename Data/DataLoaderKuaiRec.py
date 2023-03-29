import os
import numpy as np
from Data import filepaths as fp
import pandas as pd
import time
from itertools import islice
import re
from sklearn.utils import shuffle

# File path
base_path = fp.KuaiRec.ORIGINAL_DIR
mat = os.path.join(base_path, 'matrix.csv')
time_mat = os.path.join(base_path, 'time_matrix.csv')
train_path = os.path.join(base_path, 'train.csv')
test_path = os.path.join(base_path, 'test.csv')
user_feat = os.path.join(base_path, 'user_features.csv')
item_categories = os.path.join(base_path, 'item_categories.csv')
item_feat = os.path.join(base_path, 'item_daily_features.csv')
item_feat_1 = os.path.join(base_path, 'item_daily_features_1.csv')


# get time_df index
def get_time_df_time_index(begin):
    years = set()
    months = set()
    days = set()
    weeks = set()
    times = set()
    time_week_dict = {}

    data = pd.read_csv(time_mat, sep=',', index_col=0)
    data = data.drop_duplicates(subset='date', keep='first', inplace=False)
    data = data.reset_index(drop=True)

    for i in range(len(data['timestamp'])):
        time_local = time.localtime(data['timestamp'][i])
        years.add(int(time_local.tm_year))
        months.add(int(time_local.tm_mon))
        days.add(int(time_local.tm_mday))
        weeks.add(int(time_local.tm_wday + 1))
        times.add(int(time.strftime('%Y%m%d', time_local)))
        time_week_dict[int(time.strftime('%Y%m%d', time_local))] = int(time_local.tm_wday + 1)

    years = sorted(years)
    months = sorted(months)
    days = sorted(days)
    weeks = sorted(weeks)
    times = sorted(times)

    years_dict = {k: v + 0 for v, k in enumerate(years)}
    months_dict = {k: v + len(years) for v, k in enumerate(months)}
    days_dict = {k: v + (len(years) + len(months)) for v, k in enumerate(days)}
    weeks_dict = {k: v + (len(years) + len(months) + len(days)) for v, k in enumerate(weeks)}
    return years_dict, months_dict, days_dict, weeks_dict, times, time_week_dict


# generate time_df
def generate_time_df():
    indexes = []
    all_times_indexes = []
    years_dict, months_dict, days_dict, weeks_dict, times, time_week_dict = get_time_df_time_index(0)

    with open(mat, 'r', encoding='ISO-8859-1') as f:
        for line in islice(f, 1, None):
            times_indexes = []
            d = line.strip().split(',')[2]
            indexes.append(int(float(d)))
            times_indexes.extend([years_dict[int(d[:4])], months_dict[int(d[4:6])], days_dict[int(d[6:8])],
                                  weeks_dict[time_week_dict[int(float(d))]]])
            all_times_indexes.append(times_indexes)

    df = pd.DataFrame(all_times_indexes, index=indexes, columns=["year", "month", "day", "week"])
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    df = df.sort_index()
    df.to_csv(fp.KuaiRec.TIME_DF_1)


# get user_df index
def get_user_feat_index(begin, i):
    feat_indexes = set()
    with open(user_feat, 'r') as f:
        for line in islice(f, 1, None):
            d = line.strip().split(',')
            feat_indexes.add(d[i])
    return {k: v + begin for v, k in enumerate(feat_indexes)}, len(feat_indexes)


# generate user_df
def generate_user_df():
    uids = []
    all_users = []
    degree_dict, begin1 = get_user_feat_index(0, 1)
    period_dict, begin2 = get_user_feat_index(begin1, 2)
    stream_dict, begin3 = get_user_feat_index(begin1 + begin2, 3)
    author_dict, begin4 = get_user_feat_index(begin1 + begin2 + begin3, 4)
    # degree_dict = {'UNKNOWN': 0, 'full_active': 1, 'middle_active': 2, 'high_active': 3}
    print("degree", degree_dict)
    print("period_dict", period_dict)
    print("stream_dict", stream_dict)
    print("author_dict", author_dict)

    with open(user_feat, 'r') as f:
        for line in islice(f, 1, None):
            user_indexes = []
            d = line.strip().split(',')
            uids.append(d[0])
            user_indexes.extend([degree_dict[d[1]], period_dict[d[2]], stream_dict[d[3]], author_dict[d[4]]])
            all_users.append(user_indexes)

    df = pd.DataFrame(all_users, index=uids,
                      columns=["user_active_degree", "is_lowactive_period", "is_live_streamer", "is_live_streamer"])
    df.to_csv(fp.KuaiRec.USER_DF_1)


# get item time index
def get_item_time_index(begin):
    times = set()
    data = pd.read_csv(item_feat)
    data['upload_dt'] = data['upload_dt'].astype('datetime64[ns]')
    data['upload_dt'] = data['upload_dt'].apply(lambda x: x.strftime('%Y%m%d'))

    for i in data["upload_dt"]:
        times.add(i)
    times = sorted(times)
    data.to_csv(item_feat_1, index=None, encoding='utf-8-sig')

    return {k: v + begin for v, k in enumerate(times)}


# generate item_df
def generate_item_df():
    iids = []
    is_same = -1
    max_len = 0
    max_value = 0
    all_feats = []

    years_dict = get_item_time_index(max_value)

    with open(item_categories, 'r') as f:
        for line in islice(f, 1, None):
            d = line.strip().split(',')
            iids.append(d[0])
            line_feats = re.findall(r'\d+', str(d[1:]))
            feats = []
            for i in line_feats:
                feats.append(int(i))
                if (max_value < int(i)):
                    max_value = int(i)
            all_feats.append(feats)
            if len(feats) > max_len:
                max_len = len(feats)

    with open(item_feat_1, 'r', encoding='utf-8-sig') as f:
        for line in islice(f, 1, None):
            d = line.strip().split(',')
            if (is_same == int(d[0])):
                continue
            all_feats[int(d[0])].append(years_dict[d[4]])
            is_same = int(d[0])

    n_all = []
    for item in all_feats:
        n_all.append(np.random.choice(item, size=max_len + 1, replace=True))
    df = pd.DataFrame(n_all, index=iids)
    df.to_csv(fp.KuaiRec.ITEM_DF_1)


# generate quaternions
def __read_rating_four_data(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split(',')
            triples.append([int(d[0]), int(d[1]), int(d[3]), int(float(d[2]))])
    return triples


# Read user item time data
def read_data_user_item_time_df():
    user_df = pd.read_csv(fp.KuaiRec.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.KuaiRec.ITEM_DF, index_col=0)
    time_df = pd.read_csv(fp.KuaiRec.TIME_DF, index_col=0)

    # Generate user item rating time quaternions
    train_triples = __read_rating_four_data(train_path)
    test_triples = __read_rating_four_data(test_path)

    return train_triples, test_triples, user_df, item_df, time_df, max(user_df.max()) + 1, \
           max(item_df.max()) + 1, max(time_df.max()) + 1


def read_data_new():
    user_df = pd.read_csv(fp.KuaiRec.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.KuaiRec.ITEM_DF, index_col=0)
    time_df = pd.read_csv(fp.KuaiRec.TIME_DF, index_col=0)

    # Create user item rating triads
    train_triples = __read_rating_four_data(train_path)
    test_triples = __read_rating_four_data(test_path)

    return train_triples, test_triples, user_df, item_df, time_df, max(item_df.max()) + 1


# Generate triples
def __read_rating_three_data(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split(',')
            triples.append([int(d[0]), int(d[1]), int(d[3])])
    return triples


# Read user item data
def read_data_user_item_df():
    user_df = pd.read_csv(fp.KuaiRec.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.KuaiRec.ITEM_DF, index_col=0)

    train_triples = __read_rating_three_data(train_path)
    test_triples = __read_rating_three_data(test_path)

    return train_triples, test_triples, user_df, item_df, max(user_df.max()) + 1, max(item_df.max()) + 1


def read_data():
    user_df = pd.read_csv(fp.KuaiRec.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.KuaiRec.ITEM_DF, index_col=0)

    train_triples = __read_rating_three_data(train_path)
    test_triples = __read_rating_three_data(test_path)

    return train_triples, test_triples, user_df, item_df, max(item_df.max()) + 1


if __name__ == '__main__':
    pass
