from collections import Counter
import re

counts = Counter()
with open('/root/KG/DeepKE/example/ner/standard/w2ner/data/train.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tag = line.split()[-1]
        m = re.match(r'[BIO]-(.+)', tag)
        if m:
            counts[m.group(1).lower()] += 1
        else:
            counts['o'] += 1

print("标签计数：", counts)

# 计算权重（出现越少权重越大）
total = sum(counts.values())
weights = {k: total/v for k, v in counts.items()}
print("建议权重：", weights)