from transformers import BertModel

model=BertModel.from_pretrained('bert-based-uncased')

len(model.encoder.layer)
 