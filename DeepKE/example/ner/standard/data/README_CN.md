<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/data/README.md">English</a> | 简体中文 </b>
</p>

支持三种类型文件格式，包含json格式、docx格式以及txt格式。其中各个文件的格式已提供，按照所给example.*对应格式提供数据。如非txt文件，需要用到[transform_data.py](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/transform_data.py)中的json2txt函数以及doc2txt函数将其进行转换。

数据文件分别命名为train.txt,test.txt,*(valid.txt),放入data目录下,注意是valid,需要将上一步的打标签的dev改成valid
实体的类型标签在配置文件conf/train.yaml中修改