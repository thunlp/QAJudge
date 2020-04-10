# QAJudge

Code and dataset of AAAI2020 Paper **Iteratively Questioning and Answering for Interpretable Legal Judgment Prediction**.

Fork from & See frame doc at [https://github.com/haoxizhong/pytorch-worker](https://github.com/haoxizhong/pytorch-worker).

## Citation

Please cite our paper if you find it helpful.

```bibtex
@inproceedings{zhong2020iteratively,
    title={Iteratively Questioning and Answering for Interpretable Legal Judgment Prediction},
    author={Zhong, Haoxi and Wang, Yuzhong and Tu, Cunchao and Zhang, Tianyang and Liu, Zhiyuan and Sun, Maosong},
    booktitle = "Proceedings of AAAI",
    year = "2020"
}
```

## Specific params for QAJudge

`config/sample_qajudge.config` is a sample config file for QAJudge.

Notice that some general parameters may be missing in this config.

**[data]:**

- ``train_formatter_type,valid_formatter_type,test_formatter_type``: Use `ZMDqn` for crime prediction; Use `FTDqn` for article prediction.
- ``task``: Use `ft` for article prediction (unrequired for others).

**[model]:**

- ``model_name``: Use `ZMDqn`.

**[rl]:**

- ``batch_size``: Required. Size of mini-batch.
- ``epsilone``: Required. Param epsilon for greedy strategy.
- ``gamma``: Required. Discount factor.
- ``target_update``: Required. Period to update the target.
- ``memory_capacity``: Required. Capacity of memory.
- ``n_actions``: Required. Number of actions, that is, the size of question list.
- ``n_questions``: Required. Chance to question, mentioned as K in the paper.

**[ml]:**

- ``lgb_path``: The path of Predict Net model. You should put your Predict Net model named `predict_net.pkl` here.
