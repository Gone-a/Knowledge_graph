import os
import sys
import torch
import logging
import hydra
from hydra import utils
from deepke.relation_extraction.standard.tools import Serializer
from deepke.relation_extraction.standard.tools import _serialize_sentence, _convert_tokens_into_index, _add_pos_seq, _handle_relation_data , _lm_serialize
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from deepke.relation_extraction.standard.utils import load_pkl, load_csv
import deepke.relation_extraction.standard.models as models

logger = logging.getLogger(__name__)
#处理警告信息
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()


@hydra.main(config_path='/root/KG/DeepKE/example/re/standard/conf/config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    # cwd = cwd[0:-5]
    cfg.cwd = cwd
    cfg.pos_size = 2 * cfg.pos_limit + 2
    print(cfg.pretty())

    #获得预测实例
    instances = _get_predict_instance(cfg)


    #预处理数据
    data, rels = _preprocess_data(instances, cfg)
    #1.模型类型映射表
    __Model__={
        'lm':models.LM,
        'cnn':models.PCNN,
        #其他模型..
    }

    #2.选择设备
    device = torch.device('cuda' if cfg.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"使用设备:{device}")

    #3.初始化模型
    model =__Model__[cfg.model_name](cfg)
    print(f"初始化模型:{type(model).__name__}")

    #4.加载权重
    try:
        model.load(cfg.fp,device=device)
        print(f"模型权重加载成功: {cfg.fp}")
    except FileNotFoundError:
        print(f"模型权重文件未找到: {cfg.fp}")

    #5.移至设备
    model.to(device)


    #6.设置评估模式
    model.eval()
    print("模型已进入评估模式,准备预测")
    
    def process_single_piece(model, piece, device, rels):
        with torch.no_grad():
            for key in piece.keys():
                piece[key] = piece[key].to(device)
            y_pred = model(piece)
            y_pred = torch.softmax(y_pred, dim=-1)[0]  
            prob = y_pred.max().item()
            index = y_pred.argmax().item()
            if index >= len(rels):
                print("The index {} is out of range for 'rels' with length {}.".format(index, len(rels)))
                return [], 0, 0
            prob_rel = list(rels.keys())[index]
            return prob_rel, prob, y_pred
        

    # ==== 预测 ====
    # 分片处理
    max_prob = -1
    best_relation = ''
    max_len = 512  # 根据模型限制设置
    for j in range(len(data)):
        tokenized_input = data[j]['token2idx']
    
        num_pieces = len(tokenized_input) // max_len + (1 if len(tokenized_input) % max_len > 0 else 0)
        
        for i in range(num_pieces):
            start_idx = i * max_len
            end_idx = min((i + 1) * max_len, len(tokenized_input))
            current_piece_input = {'word': torch.tensor([tokenized_input[start_idx:end_idx] + [0] * (max_len - (end_idx - start_idx))]),
                                'lens': torch.tensor([min(end_idx - start_idx, max_len)])}
            relation, prob, y_pred  = process_single_piece(model, current_piece_input, device, rels)
            if prob > max_prob:
                max_prob = prob
                best_relation = relation
        logger.info(f"\"{data[0]['head']}\" 和 \"{data[0]['tail']}\" 在句中关系为：\"{best_relation}\"，置信度为{max_prob:.2f}。")

    if cfg.predict_plot:
        # maplot 默认显示不支持中文
        plt.rcParams["font.family"] = 'Arial Unicode MS'
        x = list(rels.keys())
        height = list(y_pred.cpu().numpy())
        plt.bar(x, height)
        for x, y in zip(x, height):
            plt.text(x, y, '%.2f' % y, ha="center", va="bottom")
        plt.xlabel('关系')
        plt.ylabel('置信度')
        plt.xticks(rotation=315)
        plt.show()


def _preprocess_data(data, cfg):
    
    relation_data = load_csv(os.path.join(cfg.cwd, cfg.data_path, 'relation.csv'), verbose=False)
    rels = _handle_relation_data(relation_data)
    _lm_serialize(data,cfg)

    return data, rels

def _get_predict_instance(cfg):
    instances=[]
    test_file= "/root/KG/DeepKE/example/re/standard/data/my_origin/test.csv"
    with open(test_file,'r', encoding='utf-8') as f:
        lines =f.readlines()
        for line in lines:
            if line.strip():
                sentence,head,tail,head_type,tail_type=line.strip().split(',')
                instance = dict()
                instance["sentence"] = sentence.strip()
                instance["head"] = head.strip()
                instance["tail"] = tail.strip()
                if head_type.strip() == '' or tail_type.strip() == '':
                    cfg.replace_entity_with_type = False
                    instance['head_type'] = 'None'
                    instance['tail_type'] = 'None'
                else:
                    instance['head_type'] = head_type.strip()
                    instance['tail_type'] = tail_type.strip()
                instances.append(instance)
    return instances

    

if __name__ == '__main__':
    main()