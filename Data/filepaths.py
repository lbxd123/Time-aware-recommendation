import os

ROOT = os.path.split(os.path.realpath(__file__))[0]
Model_Dir = os.path.join(ROOT,'model')


class KuaiRec():
    ORIGINAL_DIR = os.path.join(ROOT, 'KuaiRec')
    TIME_DF = os.path.join(ORIGINAL_DIR, 'time_df')
    TIME_DF_1 = os.path.join(ORIGINAL_DIR, 'time_df_1.csv')
    USER_DF = os.path.join(ORIGINAL_DIR, 'user_df')
    USER_DF_1 = os.path.join(ORIGINAL_DIR, 'user_df_1.csv')
    ITEM_DF = os.path.join(ORIGINAL_DIR, 'item_df')
    ITEM_DF_1 = os.path.join(ORIGINAL_DIR, 'item_df_1.csv')


class Ml_100K():
    ORGINAL_DIR = os.path.join(ROOT,'MovieLens')
    USER_DF = os.path.join(ORGINAL_DIR,'user_df.csv')
    USER_DF_1 = os.path.join(ORGINAL_DIR,'user_df_1.csv')
    ITEM_DF = os.path.join(ORGINAL_DIR,'item_df.csv')
    ITEM_DF_0 = os.path.join(ORGINAL_DIR, 'item_df_0.csv')
    ITEM_DF_1 = os.path.join(ORGINAL_DIR, 'item_df_1.csv')
    TIME_DF = os.path.join(ORGINAL_DIR, 'time_df.csv')
    TIME_DF_1 = os.path.join(ORGINAL_DIR, 'time_df_1.csv')


if __name__ == '__main__':
    pass


