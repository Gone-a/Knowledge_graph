from deepke.models import TransE
model = TransE.load_pretrained('path_to_pretrained_model')
predictions = model.predict(data)