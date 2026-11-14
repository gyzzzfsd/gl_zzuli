"""评估包。

放在仓库根目录（与 backend 平级），通过 `eval/_bootstrap.py` 把 backend 加入 sys.path，
从而复用生产链路的 app.* 模块，保证"评估跑的就是线上那条链路"。

目录结构：
    eval/eval_doc/qa.csv      黄金集（唯一事实来源）
    eval/run_eval.py          命令行入口
    eval/runner.py            跑图 + 采集 answer/contexts
    eval/metrics_ext.py       确定性指标（数值/拒答/引用）
    eval/ragas_adapter.py     RAGAS 五项指标（可选依赖）
    eval/report.py            报告（CSV + Markdown）
    eval/gate.py              回归门禁
    eval/selfcheck.py         无外部依赖的自检
"""