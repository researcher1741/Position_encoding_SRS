import copy
import os
import pickle
import time
import sys
from pathlib import Path
import random
import torch.multiprocessing as mp
import torch
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from src.config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY, CHECK, RESULTS
from src.config import Men_Args, Beauty_Args, Fashion_Args, Games_Args
from src.data_preprocessing import data_loading_and_partition
from src.sampler import PytorchSampler, Val_sampler
from src.torch_modules import Skeleton


class SeqRS_Trainer:
    """ Evaluation for the main_dataset set
    This creates an example for validation or testing.
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

    def __init__(self, rank, device, dataname, base_name="", seed=42, config=None, world_size=1):
        # Basic initializations
        print(f"rank {rank}, world_size {world_size}")
        self.device = torch.device(device, rank)
        self.rank, self.seed, self.world_size = rank, seed, world_size
        dist.init_process_group(backend='nccl',
                                init_method='tcp://localhost:23456',
                                world_size=world_size, rank=rank)
        self.dataname = dataname if config is None else config.dataset
        self.is_parallel = False if world_size == 1 else True
        base_name += "_" + str(seed) + "_" + self.dataname.lower()
        sys.stdout = open(os.path.join(Path(__file__).parents[1].__str__(),
                                       base_name + ".txt"), 'w')

        # The following line loads also the configuration and it can be used in what follows
        config, dataset, CXTDict, ItemFeatures = self.loading(config)

        # Metrics setting up
        self.metrics_val = dict(zip([f'NDCG@{self.top_k}', f'HIT@{self.top_k}'], [[], []]))
        self.metrics_test = dict(zip([f'NDCG@{self.top_k}', f'HIT@{self.top_k}'], [[], []]))
        self.metrics_train = {'Loss': []}

        # Model, data, optimizer
        self.prepare_data(dataset, CXTDict)
        self.prepare_model(config, ItemFeatures)
        self.get_optimizer()

        # Model name
        Tostr = lambda x: "=1" if x else "=0"
        model_name = base_name + "_" + f'R{Tostr(self.reverse)}_' + f'MaskU{Tostr(self.mask_user)}_'
        model_name += f'Samp{Tostr(self.sampling_mode)}_'
        model_name += f'OFinals{Tostr(self.only_finals)}_'
        self.model_name = model_name
        self.training()

    def loading(self, config=None):
        dataset, num_batch, config, ItemFeatures, CXTDict, UserFeatures = data_loading_and_partition(self.dataname,
                                                                                                     config)
        self.config = config
        self.print_args(config)

        # Load Configuration
        [_, _, _, unum, inum] = copy.deepcopy(dataset)
        print("Loading Configuration ...")
        self.config._add(**{"n_users": unum, "n_items": inum})
        return self.config, dataset, CXTDict, ItemFeatures

    def print_args(self, args):
        print("\n#######  Training configuration")
        for k, v in args.__dict__.items():
            if "__" not in k:
                k_print = str(k) + ":" + " " * max(0, 20 - len(k))
                print(f"{k_print} \t{v}")
                if "logger" not in k:
                    setattr(self, k, args.__dict__[k])
        print("\n#######")

    def prepare_model(self, config, Features):  # , feat_dim):
        print("Loading Model ...")
        self.model = Skeleton(config, Features, self.device)  # , feat_dim)
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Amount of model parameters", params)

    def get_optimizer(self):
        print("Loading scheduler and optimizer ...")
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=300,
                                                         num_training_steps=int(self.steps_E * self.num_epochs))

    def prepare_data(self, dataset, cxtdict):
        [train, val, test, unum, inum] = copy.deepcopy(dataset)
        assert unum == len(train) == len(val) == len(test)
        self.steps_E = unum // self.batch_size
        if isinstance(self.validation_point, float):
            self.validation_point = int(self.steps_E * self.num_epochs * self.validation_point)
        else:
            self.validation_point = int(self.steps_E * self.validation_point)
        # Other hyperparameters
        self.total_iterations = int(self.num_epochs * unum)

        Train = PytorchSampler(unum, inum, train,
                               cxtdict, self.config, device=self.device, SEED=self.seed,
                               reverse=self.reverse, mask_user=self.mask_user,
                               only_finals=self.only_finals, sampling_mode=self.sampling_mode)
        Val = Val_sampler(self.device, dataset, self.config, cxtdict, mode="validation",
                          reverse=self.reverse, size=self.test_size, seed=self.seed)
        Test = Val_sampler(self.device, dataset, self.config, cxtdict, mode="test",
                           reverse=self.reverse, size=self.test_size, seed=self.seed)
        Val.device = "cpu"
        Test.device = "cpu"
        Train.device = "cpu"

        # self.Val = DataLoader(Val, batch_size=self.batch_size, num_workers=0)
        # self.Test = DataLoader(Test, batch_size=self.batch_size, num_workers=0)
        # Train.device = "cpu"
        # self.Train = DataLoader(Train, batch_size=self.batch_size, num_workers=0)
        Val_samp = torch.utils.data.distributed.DistributedSampler(Val,
                                                                   num_replicas=self.world_size,
                                                                   rank=self.rank)
        Test_samp = torch.utils.data.distributed.DistributedSampler(Test,
                                                                    num_replicas=self.world_size,
                                                                    rank=self.rank)
        Train_samp = torch.utils.data.distributed.DistributedSampler(Train,
                                                                     num_replicas=self.world_size,
                                                                     rank=self.rank)
        self.Train = DataLoader(Train, batch_size=self.batch_size, num_workers=0, sampler=Train_samp)
        self.Val = DataLoader(Val, batch_size=self.batch_size, num_workers=0, sampler=Val_samp)
        self.Test = DataLoader(Test, batch_size=self.batch_size, num_workers=0, sampler=Test_samp)
        print(f"Number of steps in the Train dataset: {len(self.Train)}")
        print(f"Number of steps in the Validation dataset: {len(self.Val)}")
        print(f"Number of steps in the Test dataset: {len(self.Test)}")

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def save_state_dict(self, state_dict, path, filename):
        torch.save(state_dict, os.path.join(path, filename))

    def training(self):
        torch.cuda.empty_cache()
        counter, steps, final_loss, T, t0, epochs_list = 0, 0, 0, 0.0, time.time(), []
        self.model.train()
        print(f"Evaluation every {self.validation_point} steps, i.e, {self.validation_point / self.steps_E} epochs")

        pbar1 = tqdm(self.Train, total=len(self.Train), ncols=70, leave=False, unit='batch')
        pbar1.set_description("Training ...")
        for batch in pbar1:
            steps += 1
            train_sample = [value.to(self.device) for value in batch]
            u, seq, pos, neg, seqcxt, poscxt, negcxt = train_sample

            # Initialize the gradients with zeros
            self.optimizer.zero_grad()
            self.model = self.model.float()

            # Forward Propagation and Backward propagation
            loss = self.model(user_seq=u,
                              seq=seq, seqcxt=seqcxt, pos=pos,
                              neg=neg, poscxt=poscxt, negcxt=negcxt,
                              )
            loss.backward()

            # Updates
            self.optimizer.step()
            self.scheduler.step()

            # accumulate the loss for the BP
            final_loss += loss.item()

            if steps % self.validation_point == 0:
                counter += 1
                epoch = steps // self.steps_E
                epochs_list.append(epoch)
                step = steps - epoch * self.steps_E
                t1 = time.time() - t0
                T += t1
                counter = self.validation_step(step, steps, counter, epochs_list, epoch, loss)
                self.model.train()
        counter += 1
        epoch = steps // self.steps_E
        epochs_list.append(epoch)
        step = steps - epoch * self.steps_E
        t1 = time.time() - t0
        T += t1
        _ = self.validation_step(step, steps, counter, epochs_list, epoch, loss)
        print(f"Done: it took {T}")
        print(f"max value of NDCG: {max(self.metrics_test[f'NDCG@{self.top_k}'])}")
        print(f"max value of HIT: {max(self.metrics_test[f'HIT@{self.top_k}'])}")
        print("\nAfter 20 validations")
        print(f"max value of NDCG: {max(self.metrics_test[f'NDCG@{self.top_k}'][21:])}")
        print(f"max value of HIT: {max(self.metrics_test[f'HIT@{self.top_k}'][21:])}")

    def validation_step(self, step, steps, counter, epochs_list, epoch, loss):
        self.validation_case(self.Test, "test")
        self.validation_case(self.Val, "val")
        self.metrics_train['Loss'].append(loss.cpu().detach().numpy())
        print(f"Epoch: {epoch}, plus {step} steps train_loss: {loss:.4}")
        if steps % (self.validation_point * self.print_every_n_point) == 0:
            test_list = self.metrics_test[f'NDCG@{self.top_k}']
            if self.saving and test_list[-1] == test_list.max():
                check_name = f"ARS_E{epoch}_Step{step}_{self.model_name}"
                state_dict = self._create_state_dict()
                self.save_state_dict(state_dict, CHECK, check_name)
            self.print_results(self.metrics_val, epochs_list, "Val_")
            self.print_results(self.metrics_test, epochs_list, "Test_")
            self.print_results(self.metrics_train, epochs_list, "Train_")
        if self.exponential_print and counter == 10 and epoch < 1000:
            counter = 0
            self.validation_point *= 2
        return counter

    def validation_case(self, dataset, text):
        self.model.eval()
        total_acu, total_ndcg, total_hits, length = 0, 0, 0, len(dataset)
        for batch in dataset:
            u, seq, seqcxt, test_seq, testitemscxt = batch
            score = self.predict(self.model, u, seq, seqcxt, test_seq, testitemscxt)
            NDCG, HIT = self.metrics(score=score.cpu().detach().numpy(), top_k=self.top_k)  # [B, L]
            total_hits += HIT
            total_ndcg += NDCG
        total_ndcg /= length
        total_hits /= length
        print("\n#### {} Acc: {}, NDCG: {} HIT: {}".format(text, total_acu, total_ndcg, total_hits))
        if text == "test":
            self.metrics_test[f'NDCG@{self.top_k}'].append(total_ndcg)
            self.metrics_test[f'HIT@{self.top_k}'].append(total_hits)
        elif "val" in text:
            # self.metrics_val[f'Acc@{self.top_k}'].append(total_acc)
            self.metrics_val[f'NDCG@{self.top_k}'].append(total_ndcg)
            self.metrics_val[f'HIT@{self.top_k}'].append(total_hits)
        self.model.train()

    def predict(self, model, u, seq, seqcxt, test_seq, testitemscxt):
        model.eval()
        inputs = u, seq, seqcxt, test_seq, testitemscxt
        u, seq, seqcxt, test_seq, testitemscxt = [x.clone().detach().long().to(self.device) for x in inputs]
        scores = model(user_seq=u, seq=seq, seqcxt=seqcxt, test=test_seq, testcxt=testitemscxt)
        return scores

    def print_results(self, mydict, epochs_list, string):
        string += self.model_name if string[-1] == "_" else "_" + self.model_name
        if not os.path.exists(RESULTS):
            os.makedirs(RESULTS)
        with open(os.path.join(RESULTS, string + '.pkl'), 'wb') as handle:
            mydict_to_save = mydict.copy()
            mydict_to_save.update({"epochs": epochs_list})
            pickle.dump(mydict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Show/save figure as desired.
        plt.figure()
        plt.xlabel('EPOCHS')
        plt.ylabel("")
        for k in mydict.keys():
            if len(mydict[k]) > 0 and k != "confusion":
                plt.plot(epochs_list, mydict[k], label=k)
        plt.title(string)
        plt.legend()
        plt.show()
        plt.tight_layout()
        plt.savefig(string + '.png')
        plt.close()

    @staticmethod
    def metrics(score, top_k):
        NDCG, HIT = 0.0, 0.0
        B = score.shape[0]
        # take the maximum of the negatives, i.e., the minimum, and then again to get the ranking value
        rank_of_the_first_item = np.argsort(np.argsort(-score, axis=-1), axis=-1)[:, 0]
        for i in rank_of_the_first_item:
            if i < top_k:
                NDCG += 1 / np.log2(i + 2)
                HIT += 1
        HIT /= B
        NDCG /= B
        return NDCG, HIT


def callme(hidden_act, encoding, max_norm, norm_type, dataset, need):
    seeds = [102, 203, 304, 405, 506, 607, 708, 808, 901, 1001, 987, 8765, 7654, 6543][:need]
    RMHA_encoder = False
    RMHA_decoder = False
    ROPEMHA_encoder = False
    Longer = False
    print(encoding)

    if dataset == "men":
        config1 = copy.copy(Men_Args)
    if dataset == "beauty":
        config1 = copy.copy(Beauty_Args)
    if dataset == "fashion":
        config1 = copy.copy(Fashion_Args)
    if dataset == "games":
        config1 = copy.copy(Games_Args)

    if "leaky" in hidden_act:
        hidden_act = "leakyrelu"

    if "long" in encoding.lower():
        encoding = encoding.replace('long', "")
        encoding = encoding.replace('Long', "")
        Longer = True

    if "carca" == encoding:
        base_name1 = "carca_"
        positional_encoding_type = ""
        position_concatenation = False

    elif "con" == encoding:
        base_name1 = "con_"
        positional_encoding_type = "absolute"
        position_concatenation = True
    elif "nocon" == encoding:
        base_name1 = "nocon_"
        positional_encoding_type = "absolute"
        position_concatenation = False

    elif "conlearnt" == encoding:
        base_name1 = "conlearnt_"
        positional_encoding_type = "learnt"
        position_concatenation = True
    elif "learnt" == encoding:
        base_name1 = "learnt_"
        positional_encoding_type = "learnt"
        position_concatenation = False

    elif "conROPE" == encoding:
        base_name1 = "conROPE_"
        positional_encoding_type = "rope"
        position_concatenation = True
    elif "ROPE" == encoding:
        base_name1 = "ROPE_"
        positional_encoding_type = "rope"
        position_concatenation = False

    elif encoding in ["ROPE1", "ROPEMHAONE", "rope1"]:
        base_name1 = "ROPEMHAONE"
        positional_encoding_type = "rope1"
        position_concatenation = False
        ROPEMHA_encoder = False
        RMHA_encoder = False
        RMHA_decoder = False

    elif "ROPEMHA" == encoding:
        base_name1 = "ROPEMHA_"
        positional_encoding_type = ""
        position_concatenation = False
        ROPEMHA_encoder = True
        RMHA_encoder = False
        RMHA_decoder = False

    elif "RMHA4" == encoding:
        base_name1 = "RMHA4_"
        positional_encoding_type = ""
        position_concatenation = False
        RMHA_encoder = True
        RMHA_decoder = False

    for seed in seeds:
        base_name = base_name1 + "nmax_" + str(max_norm)
        base_name += "_ntype_" + str(norm_type)
        base_name += "_hact_" + str(hidden_act)
        config = copy.deepcopy(config1)
        if Longer:
            num_epochs = config.num_epochs + 400
            base_name = "Longer" + base_name
        else:
            num_epochs = config.num_epochs
        config._add(**{
            "num_epochs": num_epochs,
            "hidden_act": hidden_act,
            "ROPEMHA_encoder": ROPEMHA_encoder,
            "RMHA_encoder": RMHA_encoder,
            "RMHA_decoder": RMHA_decoder,
            "positional_encoding_type": positional_encoding_type,
            "position_concatenation": position_concatenation,
            "max_norm": max_norm,
            "norm_type": norm_type,
        })
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        world_size = 1
        mp.spawn(SeqRS_Trainer, args=("cuda",
                                      config.dataset,
                                      base_name,
                                      seed,
                                      config,
                                      world_size,),
                 nprocs=world_size, join=True)
