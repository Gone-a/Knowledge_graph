import torch
from deepke.re.model import Bert_RE  # DeepKE的关系抽取模型
from transformers import BertTokenizer  # 用于文本分词
import re  # 简单处理文本

# --------------------------
# 1. 配置参数（根据你的模型修改）
# --------------------------
class Config:
    def __init__(self):
        # 预训练模型（要和训练时用的一致，比如bert-base-chinese）
        self.model_name_or_path = "bert-base-chinese"
        # 关系类别数量（必须和训练时的数据集一致，比如你的数据有10种关系就填10）
        self.num_labels = 10  # ←这里需要你根据自己的数据集修改！
        # 最大文本长度（和训练时保持一致，一般128或256）
        self.max_seq_length = 128
        # 你的模型路径（替换成你提供的路径）
        self.model_path = "/root/KG/DeepKE/example/re/standard/checkpoints/2025-07-06_18-26-42/lm_epoch50.pth"
        # 设备（自动选择GPU或CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 2. 加载模型和分词器
# --------------------------
def load_model_and_tokenizer(config):
    # 加载分词器（将文本转成模型能理解的数字）
    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
    
    # 加载模型结构
    model = Bert_RE(
        pretrained_model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        dropout_prob=0.1  # 和训练时保持一致
    )
    
    # 加载训练好的权重（关键：加载你训练好的模型）
    model.load_state_dict(
        torch.load(config.model_path, map_location=config.device)  # 自动适配设备
    )
    
    # 切换到推理模式（关闭训练时的dropout等）
    model.to(config.device)
    model.eval()
    
    return model, tokenizer

# --------------------------
# 3. 处理输入文本（转成模型能接受的格式）
# --------------------------
def process_text(text, entity1, entity2, tokenizer, config):
    """
    输入：文本、实体1、实体2
    输出：模型需要的输入格式（tensor）
    """
    # 1. 分词（将文本拆成小词）
    tokens = tokenizer.tokenize(text)
    
    # 2. 加上特殊符号（BERT模型要求）
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    
    # 3. 处理长度（超过最大长度则截断）
    if len(tokens) > config.max_seq_length:
        tokens = tokens[:config.max_seq_length]
    
    # 4. 转成数字ID（模型只认数字）
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 5. 填充到最大长度（不足则补0）
    padding_length = config.max_seq_length - len(input_ids)
    input_ids += [0] * padding_length  # 补0
    
    # 6. 注意力掩码（告诉模型哪些是真实文本，哪些是填充的0）
    attention_mask = [1] * len(tokens) + [0] * padding_length
    
    # 7. 实体位置标记（简化版：标记实体在文本中的位置）
    # 找到实体1和实体2在文本中的位置（简单匹配，实际可更精确）
    e1_mask = [0] * config.max_seq_length
    e2_mask = [0] * config.max_seq_length
    
    # 简单匹配实体位置（实际项目中可优化）
    entity1_tokens = tokenizer.tokenize(entity1)
    entity2_tokens = tokenizer.tokenize(entity2)
    for i in range(len(tokens) - len(entity1_tokens) + 1):
        if tokens[i:i+len(entity1_tokens)] == entity1_tokens:
            e1_mask[i:i+len(entity1_tokens)] = [1] * len(entity1_tokens)
    for i in range(len(tokens) - len(entity2_tokens) + 1):
        if tokens[i:i+len(entity2_tokens)] == entity2_tokens:
            e2_mask[i:i+len(entity2_tokens)] = [1] * len(entity2_tokens)
    
    # 8. 转成PyTorch张量（模型需要的格式）
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(config.device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(config.device)
    e1_mask = torch.tensor([e1_mask], dtype=torch.float).to(config.device)
    e2_mask = torch.tensor([e2_mask], dtype=torch.float).to(config.device)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "e1_mask": e1_mask,
        "e2_mask": e2_mask
    }

# --------------------------
# 4. 用模型预测关系
# --------------------------
def predict_relation(text, entity1, entity2, model, tokenizer, config, id2label):
    """
    输入：文本、实体1、实体2、模型等
    输出：实体1和实体2的关系
    """
    # 处理文本格式
    inputs = process_text(text, entity1, entity2, tokenizer, config)
    
    # 模型预测（关闭梯度计算，加快速度）
    with torch.no_grad():
        outputs = model(** inputs)  # 输入模型
        logits = outputs[0]  # 预测结果（分数）
        pred_id = torch.argmax(logits, dim=1).item()  # 取分数最高的类别
    
    # 转成关系名称（比如0→"无关系"，1→"父亲"）
    return id2label.get(pred_id, "未知关系")

# --------------------------
# 主函数（整合所有步骤）
# --------------------------
def main():
    # 1. 配置参数（只需要改这里！）
    config = Config()
    # 关系映射：根据你的训练数据修改（比如你的模型训练时0代表"无关系"，1代表"同事"等）
    id2label = {
        0: "无关系",
        1: "父亲",
        2: "母亲",
        3: "同事",
        4: "朋友",
        # ... 这里要列全你训练时所有的关系！
    }
    
    # 2. 加载模型（关键：加载你训练好的模型）
    print("正在加载模型...")
    model, tokenizer = load_model_and_tokenizer(config)
    print("模型加载完成！")
    
    # 3. 测试文本（可以换成你想抽取的文本）
    test_text = "小明和小红是同事，他们都在腾讯公司工作"
    test_entity1 = "小明"
    test_entity2 = "小红"
    
    # 4. 抽取关系
    relation = predict_relation(
        text=test_text,
        entity1=test_entity1,
        entity2=test_entity2,
        model=model,
        tokenizer=tokenizer,
        config=config,
        id2label=id2label
    )
    
    # 5. 输出结果
    print(f"\n文本：{test_text}")
    print(f"实体1：{test_entity1}")
    print(f"实体2：{test_entity2}")
    print(f"预测关系：{relation}")

if __name__ == "__main__":
    main()