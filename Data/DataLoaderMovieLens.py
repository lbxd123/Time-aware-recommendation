import os
import numpy as np
from Data import filepaths as fp
import pandas as pd
import time

# file path
base_path = fp.Ml_100K.ORGINAL_DIR
train_path = os.path.join(base_path, 'train.csv')
test_path = os.path.join(base_path, 'test.csv')
user_path = os.path.join(base_path, 'u.user')
item_path = os.path.join(base_path, 'u.item')
occupation_path = os.path.join(base_path, 'u.occupation')


# Converting timestamps in training sets to time
def new_train(path):
    years = set()
    months = set()
    days = set()
    weeks = set()
    time_week_dict = {}

    data = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'time'])
    for i in range(len(data['time'])):
        time_local = time.localtime(data['time'][i])
        years.add(time_local.tm_year)
        months.add(time_local.tm_mon)
        days.add(time_local.tm_mday)
        weeks.add(time_local.tm_wday+1)
        time_week_dict[int(time.strftime('%Y%m%d', time_local))] = int(time_local.tm_wday + 1)
        data['time'][i] = time.strftime('%Y%m%d', time_local)

    df2 = pd.DataFrame(data)
    df2.to_csv(path, sep='\t', index=False, header=False)

    years = sorted(years)
    months = sorted(months)
    days = sorted(days)
    weeks = sorted(weeks)
    years_dict = {k: v + 0 for v, k in enumerate(years)}
    months_dict = {k: v + len(years) for v, k in enumerate(months)}
    days_dict = {k: v + (len(years)+ len(months))for v, k in enumerate(days)}
    weeks_dict = {k: v + (len(years)+ len(months)+ len(days)) for v, k in enumerate(weeks)}

    return years_dict, months_dict, days_dict, weeks_dict, time_week_dict


# create time_df
def generate_time_df():
    indexes = []
    all_times_indexes = []
    years_dict, months_dict, days_dict, weeks_dict, time_week_dict = new_train(train_path)

    with open(train_path, 'r', encoding='ISO-8859-1') as f:
        for line in f.readlines():
            times_indexes = []
            d = line.strip().split('\t')[3]
            indexes.append(d)
            times_indexes.extend([years_dict[int(d[:4])], months_dict[int(d[4:6])], days_dict[int(d[6:8])], weeks_dict[time_week_dict[int(float(d))]]])
            all_times_indexes.append(times_indexes)

    df = pd.DataFrame(all_times_indexes, index=indexes,columns=["year", "month", "day", "week"])
    df.to_csv(fp.Ml_100K.TIME_DF_1)


# get user age index
def __read_age_index():
    age_levels = set()

    with open(user_path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('|')
            age_level = int(d[1]) // 10
            age_levels.add(age_level)

    return len(age_levels)


# get user occupation index
def __read_occupation_index(begin):
    occupations = {}
    with open(occupation_path, 'r') as f:
        names = f.read().strip().split('\n')
    for name in names:
        occupations[name] = begin
        begin += 1
    return occupations, begin


# create user_df.csv
def generate_user_df():
    begin = __read_age_index()
    gender_dict = {'M': begin, 'F': begin + 1}
    begin += 2
    occupation_dict, begin = __read_occupation_index(begin)
    uids = []
    all_users = []

    with open(user_path, 'r') as f:
        for line in f.readlines():
            user_indexs = []
            d = line.strip().split('|')
            age = int(d[1]) // 10
            uids.append(d[0])
            user_indexs.append(age)
            user_indexs.append(gender_dict[d[2]])
            user_indexs.append(occupation_dict[d[3]])
            all_users.append(user_indexs)

    df = pd.DataFrame(all_users, index=uids, columns=['age', 'gender', 'occupation'])
    df.to_csv(fp.Ml_100K.USER_DF_1)

    return begin


# get item year index
def __get_year_index(begin):
    years = set()

    with open(item_path, 'r', encoding='ISO-8859-1') as f:
        for line in f.readlines():
            d = line.strip().split('|')
            year = d[2].split('-')
            if len(year) > 2:
                years.add(int(year[2]))

    years.add(0)
    years = sorted(years)

    return {k: v + begin for v, k in enumerate(years)}, len(years)


# create item_df.csv
def generate_item_df(begin):
    iids = []
    items = {}
    max_len = 0
    all_items = []
    years_dict, begin = __get_year_index(begin)

    with open(item_path, 'r', encoding='ISO-8859-1') as f:
        for line in f.readlines():
            item_index = []
            d = line.strip().split('|')
            iids.append(int(d[0]))
            year = d[2].split('-')
            if len(year) > 2:
                item_index.append(years_dict[int(year[2])])
            else:
                item_index.append(0)

            subjects = d[5:]
            if begin == 0:
                begin = len(subjects)
            for i in range(len(subjects)):
                if int(subjects[i]) == 1:
                    item_index.append(begin + i)
            all_items.append(item_index)
            if len(item_index) > max_len:
                max_len = len(item_index)
    n_all = []
    for item in all_items:
        n_all.append(np.random.choice(item, size=max_len, replace=True))

    df = pd.DataFrame(n_all, index=iids)
    df.to_csv(fp.Ml_100K.ITEM_DF_1)

    return items


def get1or0(r):
    return 1 if r > 3 else 0


# Generate quaternions
def __read_rating_four_data(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('\t')
            triples.append([int(d[0]), int(d[1]), get1or0(int(d[2])), int(d[3])])
    return triples

# Generate quaternions
def __read_rating_three_data(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('\t')
            triples.append([int(d[0]), int(d[1]), int(get1or0(int(d[2])))])
    return triples


# Read user item data
def read_data_user_item_df():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF_0, index_col=0)

    train_triples = __read_rating_three_data(train_path)
    test_triples = __read_rating_three_data(test_path)

    return train_triples, test_triples, user_df, item_df, max(user_df.max()) + 1, max(item_df.max()) + 1


# Read user item time data
def read_data_user_item_time_df():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF_0, index_col=0)
    time_df = pd.read_csv(fp.Ml_100K.TIME_DF, index_col=0)

    train_triples = __read_rating_four_data(train_path)
    test_triples = __read_rating_four_data(test_path)

    return train_triples, test_triples, user_df, item_df, time_df, max(user_df.max()) + 1, max(item_df.max()) + 1, max(time_df.max()) + 1


def read_data():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF, index_col=0)
    train_triples = __read_rating_three_data(train_path)
    test_triples = __read_rating_three_data(test_path)
    return train_triples, test_triples, user_df, item_df, max(item_df.max()) + 1


def read_data_new():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF, index_col=0)
    time_df = pd.read_csv(fp.Ml_100K.TIME_DF, index_col=0)
    train_triples = __read_rating_four_data(train_path)
    test_triples = __read_rating_four_data(test_path)
    return train_triples, test_triples, user_df, item_df, time_df, max(item_df.max()) + 1


# drop_duplicates
def time_df_duplicates():
    df= pd.read_csv(fp.Ml_100K.TIME_DF, sep=',')
    df = df.sort_index()
    df["new"] = df.index
    df = df.drop_duplicates("new")
    df = df.set_index("new")
    df.to_csv(fp.Ml_100K.TIME_DF_1)

if __name__ == '__main__':
    pass