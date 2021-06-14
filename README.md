# LAMBERT

Code of LAMBERT's (https://arxiv.org/abs/2002.08087) PyTorch modules. They operate as wrappers over corresponding RoBERTa models from HuggingFace's transformers library. See `example.py`.

## Dataset

The `dataset` directory contains a CSV file with links to ~200k documents which comprise the dataset that was used in the ablation study in the paper. For each file additional metadata is provided:
* number of pages
* number of tokens
* binary classification into _business/legal_ and _other_ classes

The dataset is a subset of a larger set, created by extracting links to PDF files from Common Crawl dump. To reduce the probability of accumulating large clusters of similarly formatted documents, we imposed an upper limit of 10 documents per domain. For every link we picked, we downloaded the corresponding document and fed it to an OCR system, retaining only the successfully (i.e. without any errors) processed ones. Although seemingly this is not necessary for born-digital PDFs, in the end we are interested in token bounding boxes; treating all documents as images, and processing them with OCR is a unified method of obtaining these.

Having OCRed the documents, we used a language detection mechanism `langdetect` (https://pypi.org/project/langdetect/) in order to filter out all the non-English PDFs. We removed all documents whose English probability is below the threshold set to 0.9. Note that this includes PDFs which were incorrectly OCRed, in particular low quality scans or documents containing longer handwritten fragments.

A subset of ~200k documents was then randomly chosen and enriched with the classification into _business/legal_ and _other_ classes. In order to obtain this classification, we manually annotated 1000 documents, and used them to train a classifier which was subsequently applied to the remaining documents. 

