from fastapi import APIRouter
import torch
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

# class BERTDataset(Dataset):
#     def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
#                  pad, pair):
#         transform = nlp.data.BERTSentenceTransform(
#             bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

#         self.sentences = [transform([i[sent_idx]]) for i in dataset] # 문장 변환
#         self.labels = [np.int32(i[label_idx]) for i in dataset] # label 변환

#     def __getitem__(self, i):
#         return (self.sentences[i] + (self.labels[i], ))

#     def __len__(self): # 전체 데이터셋의 길이 반환
#         return (len(self.labels))

# def predict(predict_sentence): # input = 감정분류하고자 하는 sentence

#     data = [predict_sentence, '0']
#     dataset_another = [data]

#     another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False) # 토큰화한 문장
#     test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 5) # torch 형식 변환

#     loaded_model.eval()

#     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
#         token_ids = token_ids.long().to(device)
#         segment_ids = segment_ids.long().to(device)

#         valid_length = valid_length
#         label = label.long().to(device)

#         out = loaded_model(token_ids, valid_length, segment_ids)


#         test_eval = []
#         for i in out:
#             logits = i
#             logits = logits.detach().cpu().numpy()

#             if np.argmax(logits) == 0:
#                 test_eval.append("분노")
#             elif np.argmax(logits) == 1:
#                 test_eval.append("기쁨")
#             elif np.argmax(logits) == 2:
#                 test_eval.append("불안")
#             elif np.argmax(logits) == 3:
#                 test_eval.append("당황")
#             elif np.argmax(logits) == 4:
#                 test_eval.append("슬픔")
#         return (test_eval[0])


router = APIRouter(
    prefix="/model",
    tags=["items"],
    responses={404: {"description": "Not Found"}},
)

@router.get("/model")
def sentiment_analysis(query: str):
    return {"query" : predict(query)}
