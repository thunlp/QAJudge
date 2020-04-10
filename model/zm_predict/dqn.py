import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from model.zm_predict.usersim import UserSimulator
from model.zm_predict.memory import ReplayMemory
from tools.accuracy_init import init_accuracy_function
from model.loss import MultiLabelSoftmaxLoss
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertAdam
import pickle
import os


class ZMDqn(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ZMDqn, self).__init__()

        try:
            self.save_details=config.get("eval", "print_details").find("yes")!=-1
        except:
            self.save_details=False
        
        self.accuracy_function = init_accuracy_function(config, *args, **params)
        self.learning_rate = config.getfloat("train", "learning_rate")
        self.batch_size = config.getint("rl", "batch_size")
        self.epsilone = config.getfloat("rl", "epsilone")
        self.gamma = config.getfloat("rl", "gamma")
        self.target_update = config.getint("rl", "target_update")
        self.memory_capacity = config.getint("rl", "memory_capacity")
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

        self.n_actions = config.getint("rl", "n_actions")
        self.n_question = config.getint("rl", "n_questions")
        self.output_dim = config.getint("model", "output_dim")
        self.n_states = self.n_actions
        self.user = UserSimulator()

        self.agent_initialized = False

        self.memory = ReplayMemory(self.memory_capacity)
        self.policy_net = nn.Linear(self.n_states + 768, self.n_actions)
        self.target_net = nn.Linear(self.n_states + 768, self.n_actions)
        self.i_episode = 0
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
        self.criterion_out = MultiLabelSoftmaxLoss(config)

        lgb_path = config.get("ml", "lgb_path")
        self.lgb = pickle.load(open(os.path.join(lgb_path, "predict_net.pkl"), "rb"))

    def output_model(self, state_list):
        x = self.lgb.predict(state_list).tolist()
        return x

    def init_multi_gpu(self, device, config, *args, **params):
        self.policy_net = nn.DataParallel(self.policy_net, device_ids=device)
        self.target_net = nn.DataParallel(self.target_net, device_ids=device)
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def take_action(self, text, state, a):
        state[a] = self.user.answer(text, a) * 2 - 1
        return state

    def forward(self, data, config, gpu_list, acc_result, mode):
        y = []
        _, after_bert = self.bert(data['token'])
        after_bert = after_bert.view(after_bert.size()[0], -1)
        is_train = mode == "train"
        for j, x in enumerate(data['input']):
            cur_state = np.zeros((self.n_states))
            state_list = [cur_state]
            action_list = []
            history_action = np.ones(self.n_actions)
            for step in range(self.n_question):
                action = self.select_action(cur_state, history_action, after_bert[j:(j + 1)], is_train=is_train)
                action_list.append(action)
                history_action[action] = 0.
                next_state = self.take_action(x, cur_state, action)
                cur_state = next_state
                state_list.append(cur_state)
            final_state = cur_state.copy()
            best_state = cur_state.copy()
            for i in range(self.n_actions):
                if final_state[i] == 0:
                    final_state[i] = -1
                    best_state = self.take_action(x, best_state, i)
            yes_unknown = (best_state.sum() - final_state.sum()) / 2
            final_result = self.output_model([final_state])[0]
            if self.save_details and (not is_train):
                temp_json={
                    "text":x,
                    "actions":[(a,final_state[a]) for a in action_list],
                    "final_state":str(final_state),
                    "best_state ":str(best_state),
                    "result":int(final_result),
                    "answer":int(data['label'][j]),
                    "yes_unknown":yes_unknown
                }
                import json
                with open("detail.log.txt","a") as temp_file:
                    temp_file.write(json.dumps(temp_json,ensure_ascii=False,indent=2))
                    temp_file.write('\n')
            reward = self.calc_reward(state_list, yes_unknown, final_result, label=data['label'][j])
            if is_train:
                for i in range(len(state_list) - 1):
                    self.memory.push(state_list[i], action_list[i], reward[i], state_list[i + 1],
                                     after_bert[j].cpu().tolist())
                self.optimize_model()
            if self.i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.i_episode += 1
            temp_y = [0 for i in range(self.output_dim)]
            temp_y[final_result] = 1
            y.append(temp_y)
        y = Variable(torch.FloatTensor(y)).cuda()
        if "label" in data.keys():
            label = data["label"]
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": 0, "acc_result": acc_result}
        return {}

    def select_action(self, x, history_action, after_bert, is_train):
        if np.random.uniform() < self.epsilone or not is_train:
            x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).cuda()
            y = torch.cat([x, after_bert], dim=1)
            action_value = self.policy_net.forward(y)
        else:  # random
            action_value = torch.rand(1, self.n_actions)
        action_value = F.softmax(action_value, dim=1).cpu() * torch.Tensor(history_action)
        action = torch.max(action_value, 1)[1].data[0]
        return int(action)

    def calc_reward(self, state_list, yes_unknown, final_result, label):
        ret = []
        n = len(state_list)
        for j in range(1, n):
            if final_result == label:
                result = 1
            else:
                result = -1
            pos = (state_list[j].sum() - state_list[j - 1].sum() + 1) / 2
            penalty = 0
            if (pos == 0):
                penalty = 1 - (0.5 ** yes_unknown)

            cur_reward = result * (1 - (0.9 ** (j - 1))) * 50 + pos * 50 - penalty * 200
            ret.append(cur_reward)
        return ret

    def optimize_model(self):
        b_s_cur, b_a, b_r, b_s_nxt, b_ab = self.memory.sample(self.batch_size)
        if not b_s_cur:
            return
        b_s_cur = Variable(torch.FloatTensor(b_s_cur)).cuda()
        b_a = Variable(torch.LongTensor(b_a)).cuda().view(-1, 1)
        b_r = Variable(torch.FloatTensor(b_r)).cuda()
        b_s_nxt = Variable(torch.FloatTensor(b_s_nxt)).cuda()
        b_ab = Variable(torch.FloatTensor(b_ab)).cuda()
        av = self.policy_net(torch.cat([b_s_cur, b_ab], dim=1)).gather(1, b_a)
        sv_next = self.target_net(torch.cat([b_s_nxt, b_ab], dim=1)).detach()
        av_expected = (sv_next.max(1)[0] * self.gamma) + b_r
        av_expected = av_expected.view(-1, 1)
        loss = self.loss_function(av, av_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
