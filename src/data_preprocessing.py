from typing import List
import numpy as np
from collections import defaultdict

from src.config import FASHION_CXT, MEN_CXT, BEAUTY_CXT, GAMES_CXT
from src.config import FASHION_ITEMS, MEN_ITEMS, BEAUTY_ITEMS, GAMES_ITEMS
from src.utils import load_data
from src.config import Beauty_Args, Fashion_Args, Men_Args, Games_Args


def get_ItemData(name: str):
    """ Loading items data
    Args:
        name (str): name of the dataset
    """
    NAMES = dict(zip(['fashion', 'men', 'beauty', 'video_games'],
                     [FASHION_ITEMS, MEN_ITEMS, BEAUTY_ITEMS, GAMES_ITEMS]))
    path = NAMES[name.lower()]
    ItemFeatures = load_data(path)
    ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures))
    return ItemFeatures


def get_CXTData(name):
    """ Loading context data, this is, the interactions, the ratings
    Args:
        name (str): name of the dataset
    :return Dict[tuple(int, int): List[float, float, ...]]
    """
    NAMES = dict(zip(['fashion', 'men', 'beauty', 'video_games'],
                     [FASHION_CXT, MEN_CXT, BEAUTY_CXT, GAMES_CXT]))
    path = NAMES[name.lower()]
    return load_data(path)


def get_UserData(usernum):
    """ Loading user data """
    UserFeatures = np.identity(usernum, dtype=np.int8)
    UserFeatures = np.vstack((np.zeros(UserFeatures.shape[1], dtype=np.int8), UserFeatures))
    return UserFeatures


def get_features(args):
    """ gets all the features for users, items and cxt.
    :return
        ItemFeatures: DataFrame: Users x features
        CXTDict: Dict
        UserFeatures: List
    """
    ItemFeatures = get_ItemData(args.dataset)
    CXTDict = get_CXTData(args.dataset)
    UserFeatures = []
    print("ItemFeatures DF dimensions", ItemFeatures.shape)
    return ItemFeatures, CXTDict, UserFeatures


def data_partition(fname: str) -> List:
    """
    Split the dataset into train, valid, test. Returns the splits and the number of users and the number of items
    :param fname: 'fashion', 'men', 'beauty' or 'games'
    :return: splits of the users in format Dict[int,List[int]] : {user:List[songs]}
    """
    users_total_num, items_total_num = 0, 0
    user_train, user_valid, user_test = {}, {}, {}
    User = defaultdict(list)

    # assume user/item index starting from 1
    f = open('./Data/%s.txt' % fname, 'r')
    for line in f:
        user_ids, items_ids = line.rstrip().split(' ')
        user_ids = int(user_ids)
        items_ids = int(items_ids)
        users_total_num = max(user_ids, users_total_num)
        items_total_num = max(items_ids, items_total_num)
        User[user_ids].append(items_ids)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    # Dict[int,List[int]]
    return [user_train, user_valid, user_test, users_total_num, items_total_num]


def data_loading_and_partition(dataset_name, config=None):
    """ This function preprocess the data: split, """
    args = config
    if config is None:
        if dataset_name == 'Beauty':
            args = Beauty_Args
        elif dataset_name == 'Fashion':
            args = Fashion_Args
        elif dataset_name == 'Men':
            args = Men_Args
        elif dataset_name == 'Video_Games':
            args = Games_Args
    else:
        args = config

    # SPLITS: Dict[int,List[int]]
    [user_train, user_valid, user_test, users_total_num, items_total_num] = data_partition(args.dataset)

    # PRINTS
    num_batch = (args.num_epochs * len(user_train)) // args.batch_size
    print(" The dataset {} contains {} users and {} items in total".format(dataset_name,
                                                                           str(users_total_num),
                                                                           str(items_total_num)))

    cc = 0.0
    for user_id in user_train:
        cc += len(user_train[user_id])
    print('average sequence length: {%.2f}' % (cc / len(user_train)))
    ItemFeatures, CXTDict, UserFeatures = get_features(args)

    return [user_train, user_valid, user_test, users_total_num, items_total_num], \
           num_batch, args, ItemFeatures, CXTDict, UserFeatures
