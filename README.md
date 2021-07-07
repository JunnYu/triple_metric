# triple_metric
triple_metric

# Dependecy
- `datasets`
- `torch`
- `fastcore`

# Use
```python
import torch
from datasets import load_metric
from fastcore.foundation import L

metric = load_metric("triple_metric.py")
pred = torch.load("preds")
references = torch.load("golds")

def find_best_f1(pred,references,id2label=None,thres=0.62):
    predictions = list(L(pred).map(lambda x:L(x).filter(lambda y:y[3]>thres and y[1]<=y[2]).map(lambda z:z[:3])))
    m = metric.compute(predictions=predictions,references=references,id2label=id2label,digits=6)
    return m

max_f1 = 0.0
id2label = dict(zip(range(9),["bod", "dep", "dis", "dru", "equ", "ite", "mic", "pro", "sym"]))
for i in range(30,95):
    thres = i/100
    m = find_best_f1(pred,references,id2label=None,thres=thres)
    if m["micro_avg"].f1>=max_f1:
        max_f1 = m["micro_avg"].f1
        print("thres",thres,"\t","micro_avg",m["micro_avg"])

```