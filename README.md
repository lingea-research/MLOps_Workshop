# MLOps_Workshop

1. pip install mlflow
2. mlflow server --host 127.0.0.1 --port 8080
3. mlflow.get_artifact_uri()
3. http://localhost:8080

### References
* https://fasttext.cc/docs/en/supervised-tutorial.html

## Commands
* wc cs.txt
* head -n 750 cs.txt > cs_train.txt
* tail -n 141 cs.txt > cs_valid.txt
* cat cs_train.txt en_train.txt hu_train.txt sk_train.txt tr_train.txt > all_train.txt
* cat cs_valid.txt en_valid.txt hu_valid.txt sk_valid.txt tr_valid.txt > all_valid.txt
* ~/tools/fastText-0.9.2/fasttext supervised -input all_train.txt -output model_ver1
* ~/tools/fastText-0.9.2/fasttext predict model_ver1.bin.bin -
* ~/tools/fastText-0.9.2/fasttext test model_ver1.bin.bin all_valid.txt
* ~/tools/fastText-0.9.2/fasttext supervised -input all_train.txt -output model_ver2 -epoch 25
* ~/tools/fastText-0.9.2/fasttext supervised -input all_train.txt -output model_ver3 -epoch 25 -lr 1.0
* ~/tools/fastText-0.9.2/fasttext supervised -input all_train.txt -output model_ver6 -epoch 25 -lr 0.1 -wordNgrams 2
* 