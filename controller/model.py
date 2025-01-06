from fastapi import APIRouter
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = np.bool_
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook
from tqdm.notebook import tqdm
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

print("토크나이저 로딩 시작")
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
print("토크나이저 로딩 완료")
print("모델 로딩 시작")
bertmodel = BertModel
print("모델 로딩 완료")
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize

device = torch.device("cpu")
max_len = 64 # max seqence length
batch_size = 64
warmup_ratio = 0.1
num_epochs = 3
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset] # 문장 변환
        self.labels = [np.int32(i[label_idx]) for i in dataset] # label 변환

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self): # 전체 데이터셋의 길이 반환
        return (len(self.labels))

class BERTClassifier(nn.Module):
  def __init__(self, bert, hidden_size=768, num_classes=5, dr_rate=None, params=None):
    super(BERTClassifier, self).__init__()
    self.bert = bert
    self.dr_rate = dr_rate

    self.classifier = nn.Linear(hidden_size , num_classes)
    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)

  def gen_attention_mask(self, token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
    return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.gen_attention_mask(token_ids, valid_length)
    _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
    if self.dr_rate:
        out = self.dropout(pooler)
    else:
        out = pooler
    return self.classifier(out)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

loaded_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
loaded_model.load_state_dict(torch.load("model_state_dict.pt"))

def predict(predict_sentence): # input = 감정분류하고자 하는 sentence

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False) # 토큰화한 문장
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 5) # torch 형식 변환

    loaded_model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = loaded_model(token_ids, valid_length, segment_ids)


        test_eval = []
        for i in out: # out = model(token_ids, valid_length, segment_ids)
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("분노")
            elif np.argmax(logits) == 1:
                test_eval.append("기쁨")
            elif np.argmax(logits) == 2:
                test_eval.append("불안")
            elif np.argmax(logits) == 3:
                test_eval.append("당황")
            elif np.argmax(logits) == 4:
                test_eval.append("슬픔")
        return (test_eval[0])


router = APIRouter(
    prefix="/model",
    tags=["items"],
    responses={404: {"description": "Not Found"}},
)

@router.get("")
def sentiment_analysis(query: str):
    return {"query" : predict(query)}
