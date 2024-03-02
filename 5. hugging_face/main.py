import fasttext
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(repo_id="lingea/small_fasttext_model_for_language_detection", filename="model4.bin")
model = fasttext.load_model(model_path)
print(model.labels)

ds_file = hf_hub_download(repo_id='lingea/small_raw_dataset_for_language_detection', filename='all.txt', repo_type="dataset")
with open(ds_file) as f:
    text = f.read()
    print(text)
