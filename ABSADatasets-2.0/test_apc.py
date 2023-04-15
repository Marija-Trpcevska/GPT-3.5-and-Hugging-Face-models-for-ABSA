from pyabsa import AspectPolarityClassification as APC

dataset = '.\\integrated_datasets\\apc_datasets\\100.CustomDataset'

# find a suitable checkpoint and use the name:
sentiment_classifier = APC.SentimentClassifier(
    checkpoint=".\\checkpoints\\fast_lsa_t_v2_custom_dataset_acc_91.95_f1_67.42"
)  # here I use the english checkpoint which is trained on all English datasets in PyABSA

sentiment_classifier.batch_predict(
    target_file=dataset,  # the batch_predict() is only available for a file only, please put the examples in a file
    print_result=True,
    save_result=False,
    ignore_error=True,
    eval_batch_size=32,
)
