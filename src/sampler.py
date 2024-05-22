import copy
import torch
import numpy as np


def random_neq(l, r, s):
    """ random integer between l and r but avoiding s """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class Sampler:
    def __init__(self, users_total_num, items_total_num, user_train, maxlen, # ItemFeatures,
                 cxtdict, cxt_size, iterations_num, SEED=42, reverse=True,
                 mask_user=False, only_finals=True, sampling_mode=False):
        """
        - reverse: True, then the sequence starts from the end. False from the start
        - mask_user: True, then we don't show the user to the model, and we only show the related items.
        - only_finals: True, then we only use the last element from the list of items. If False we get a sublist.
            This adds diversity to the selection.
        - sampling_mode: True, then we select the users in order instead of sampling
        """
        # The maximum index is users_total_num + 1 because of the padding
        self.max_user = users_total_num  # + 1
        self.max_item = items_total_num  # + 1
        self.user_train, self.maxlen = user_train, maxlen
        self.cxtdict, self.cxt_size = cxtdict, cxt_size
        # self.ItemFeatures, self.att_size = ItemFeatures, ItemFeatures.shape[-1]
        self.reverse, self.mask_user, self.only_finals = reverse, mask_user, only_finals
        np.random.seed(SEED)
        self.sampling_mode = sampling_mode
        if not self.sampling_mode:  # in order
            self.iterations_num = iterations_num
        else:
            self.iterations_num = len([0 for k, v in self.user_train.items() if len(v) > 1])
            self.position = 1

    def __len__(self):
        return self.iterations_num

    def sample(self):
        """ This function creates a sample.
        Outputs if reverse:
            seq: [0,0, ... , 0, 345, 29, 1203] = [0,0, ... , 0, n-3, n-2, n-1]
            pos: [0,0, ... , 0, 29, 1203, 942] = [0,0, ... , 0, n-2, n-1, n]
            neg: [0,0, ... , 0, random, random, random]
            CXT versions: seqcxt, poscxt, negcxt ; they are associated to the items ids
        """
        # Select user
        if self.sampling_mode and self.position <= self.users_total_num:  # Pick in order the user. -> user, int
            user = self.position
            self.position += 1
        else:  # Pick randomly a user. -> user
            user = np.random.randint(1, self.max_user)
            while len(self.user_train[user]) <= 1:
                user = np.random.randint(1, self.max_user)

        # Select cutting point -> mylist
        main_list = self.user_train[user]
        maximum = len(main_list)
        if maximum > 1 and not self.only_finals:
            mylist = []
            while len(mylist) <= 1:
                start = np.random.randint(0, maximum-1)
                end = np.random.randint(start, maximum)
                mylist = main_list[start:end]
        else:
            mylist = main_list

        # Sequence Padding for user -> pos, neg, seq initialized
        seq = np.zeros([self.maxlen], dtype=np.int32)  # , dtype=np.int32)
        pos, neg = seq.copy(), seq.copy()

        # Sequence Padding for context -> poscxt, negcxt, seqcxt initialized
        seqcxt = np.zeros([self.maxlen, self.cxt_size], dtype=np.float32)  # , dtype=np.float32)
        poscxt, negcxt = seqcxt.copy(), seqcxt.copy()

        # Target and its index, which is going to be the last one
        target = mylist[-1]
        idx = self.maxlen - 1

        ts = set(mylist)
        for item_id in reversed(mylist[:-1]):
            # MAIN
            seq[idx] = item_id
            pos[idx] = target
            # random selection of examples outside of the positives
            neg_i = random_neq(1, self.max_item, ts) if target != 0 else 0
            neg[idx] = neg_i

            # CXT
            seqcxt[idx] = self.cxtdict[(user, item_id)]
            poscxt[idx] = self.cxtdict[(user, target)]
            negcxt[idx] = self.cxtdict[(user, target)]

            # LOOP UPDATE
            target = item_id
            idx -= 1
            if idx == -1: break
        if self.mask_user:
            user_seq = np.where(pos != 0, user, pos)
        else:
            user_seq = np.ones(self.maxlen) * user
        user_seq = user_seq.astype(int)

        if not self.reverse:
            user_seq, seq, pos, neg = user_seq[::-1], seq[::-1], pos[::-1], neg[::-1]
            seqcxt, poscxt, negcxt = seqcxt[::][::-1], poscxt[::][::-1], negcxt[::][::-1]
        return user_seq, seq, pos, neg, seqcxt, poscxt, negcxt

    def create_batch(self, batch_size):
        # Initialization
        return [self.sample() for i in range(batch_size)]


class PytorchSampler(Sampler):
    def __init__(self, users_total_num, items_total_num, user_train,
                 cxtdict, args, device, SEED=42,
                 reverse=True, mask_user=False, only_finals=True, sampling_mode=False):
        maxlen = args.maxlen
        iterations_num = int(args.num_epochs * len(user_train))
        super().__init__(users_total_num, items_total_num, user_train, maxlen,
                         cxtdict, args.cxt_size, iterations_num, SEED, reverse, mask_user, only_finals, sampling_mode)
        self.device = device
        # torch.manual_seed(SEED)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        tuple_output = self.sample()
        return [torch.tensor(x, device=self.device) for x in tuple_output]  # dtype=torch.short,


class Val_sampler:
    """ Evaluation for the val/test set
    This creates an example for validation or testing.
    Seq correspond to the true positive, a combination of train positives for the user
    with a last positive from the val/test dataset
    test: correponds to the negative values.
        - dataset: reference dataset, list of datasets validation/test of the form Dict[int:List[int]]
            and items and user counts
        - args: Args object with the configuration
        - unum: number of users
        - inum: number of items
        - cxtdict: Dict[tuple: List[float]]
        - metrics: (NDCG, HT, valid_user, Auc)
        - case: str, if validation then use validation as last example
        - negnum: number of negative examples to be created
    """
    # TODO: Check if the output is not shuffling the sequence
    def __init__(self, device, dataset, args, cxtdict, mode="validation",
                 reverse=True, size=10000, seed=42):
        if "val" in mode:
            [self.train_data, self.data, _, self.unum, self.inum] = copy.deepcopy(dataset)
        elif "test" in mode:
            [self.train_data, self.valid, self.data, self.unum, self.inum] = copy.deepcopy(dataset)
        # users to be used
        if self.unum > size:
            self.list_users = (self.rand_user() for _ in range(size))  # + 1
        else:
            self.list_users = range(1, self.unum)  # + 1
            size = self.unum
        self.mode = mode.lower()
        self.cxtdict = cxtdict
        self.args = args
        self.negnum = args.maxlen
        self.reverse = reverse
        self.device = device
        self.size = size
        # torch.manual_seed(seed)

    def __len__(self):
        return self.size

    def rand_user(self):
        return np.random.randint(1, self.unum)

    def sample(self, u):
        """
        :param
            u: int, user id
        :return:
            u: user
            seq: positive examples: [train1, train2, ..., trainN-1, Val_N]
            seqcxt
            test_seq: negative: [rand, rand,..., rand]
            testitemscxt
        """
        # Skip when it is too short
        # u = next(self.list_users)
        while len(self.train_data[u]) < 1 or len(self.data[u]) < 1:
            u += 1
            if u > self.unum:
                u = self.rand_user()

        # Init sequences with padding
        seq = np.zeros([self.args.maxlen], dtype=np.int32)
        seqcxt = np.zeros([self.args.maxlen, self.args.cxt_size], dtype=np.int32)  # cxt_size=6 in amazon

        # Init output sequence
        testitemscxt = list()
        idx = self.args.maxlen - 1

        # We add one item for the validation case: seq and seqcxt
        if "val" in self.mode:
            pass
        #     seq[idx] = self.data[u][0]
        #     seqcxt[idx] = self.cxtdict[(u, self.data[u][0])]
        #     idx -= 1
        # --> [0,0,0, ...,0,0,val_item]
        elif "test" in self.mode:
            seq[idx] = self.valid[u][0]
            seqcxt[idx] = self.cxtdict[(u, self.valid[u][0])]
            idx -= 1
        else:
            raise ValueError(
                f"Validation and test dataloaders must contains a mode attribute with either 'test' or 'val'"
                f" in this case we got: {self.mode})."
            )

        # We train part: seq and context
        # --> [train_item,train_item,train_item, ...,train_item,train_item,val_item]
        for i in reversed(self.train_data[u]):
            seq[idx] = i
            seqcxt[idx] = self.cxtdict[(u, i)]
            idx -= 1
            if idx == -1: break

        # Rated set
        rated = set(self.train_data[u])
        rated.add(0)

        # --> [0,0,0, ...,0,0,val_item]
        test_seq = [self.data[u][0]]
        testitemscxt.append(self.cxtdict[(u, self.data[u][0])])

        # negative examples loop/ no rated
        for _ in range(self.negnum - len(test_seq)):  # ):
            t = random_neq(1, self.inum + 1, rated)
            test_seq.append(t)
            testitemscxt.append(self.cxtdict[(u, self.data[u][0])])
        u = np.ones(self.args.maxlen) * u

        if not self.reverse:
            seq, test_seq = seq[::-1], test_seq[::-1]
            seqcxt, testitemscxt = seqcxt[::][::-1], testitemscxt[::][::-1]
        return u, seq, seqcxt, test_seq, testitemscxt

    def __getitem__(self, idx):
        tuple_output = self.sample(self.rand_user())  # idx + 1
        return [torch.tensor(x, device=self.device) for x in tuple_output]
