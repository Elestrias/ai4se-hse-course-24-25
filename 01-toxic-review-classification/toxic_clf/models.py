import pandas as pd
import warnings
import numpy as np
import pandas
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score

GSCV_parameters = [{'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
                   {'penalty': ['none', 'elasticnet', 'l1', 'l2']},
                   {'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

warnings.filterwarnings("ignore")

def classifier(dataset, model):
    # Use real X and y from dataset
    print(dataset)
    dataset = dataset.to_pandas()
    X = dataset["message"]
    y = dataset["is_toxic"]
    scores = []
    matrices = []

    x_train, x_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=46, shuffle=True)
    if model == "classic_ml":
        tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
        train_tfidf = tfidf_vec.fit_transform(x_train)
        test_tfidf = tfidf_vec.transform(x_val)
        print(tfidf_vec.get_feature_names_out()[:10])

        res_model = model_classic_train(train_tfidf, y_train)
        eval_res = eval_fun_br(test_tfidf, y_val, res_model)
        print({'tn': eval_res[1][0, 0], 'fp': eval_res[1][0, 1],
               'fn': eval_res[1][1, 0], 'tp': eval_res[1][1, 1]})


    elif model in ["roberta-base", "microsoft/codebert-base"]:
        x_tr, x_test, y_tr, y_test = train_test_split(
            x_train,
            y_train,
            test_size=0.8,
            random_state=46,
            shuffle=True)
        tokenizer = RobertaTokenizer.from_pretrained(model)
        tf_bert = lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length")

        trd = Dataset.from_pandas(pd.DataFrame({"text": x_tr, "labels": y_tr})).map(tf_bert, batched=True)
        tsd = Dataset.from_pandas(pd.DataFrame({"text": x_test, "labels": y_test})).map(tf_bert, batched=True)
        vald = Dataset.from_pandas(pd.DataFrame({"text": x_val, "labels": y_val})).map(tf_bert, batched=True)
        res_model = bert_model(dataset, trd, tsd, model)
        metrics = res_model.evaluate(vald)
        print(f"MODEL \"{model}\" VALIDATION RESULTS")
        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Accuracy: {metrics['eval_accuracy']:.3f}")
        print(f"Precision: {metrics['eval_precision']:.3f}")
        print(f"Recall: {metrics['eval_recall']:.3f}")
        print(f"F1 Score: {metrics['eval_f1']:.3f}")
        print("--------------------------")

    elif model == "all":
        tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
        train_tfidf = tfidf_vec.fit_transform(x_train)
        test_tfidf = tfidf_vec.transform(x_val)
        print(tfidf_vec.get_feature_names_out()[:10])
        res_model = model_classic_train(train_tfidf, y_train)

        x_tr, x_test, y_tr, y_test = train_test_split(
            x_train,
            y_train,
            test_size=0.8,
            random_state=46,
            shuffle=True)

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        tf_bert = lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length")

        trd = Dataset.from_pandas(pd.DataFrame({"text": x_tr, "labels": y_tr})).map(tf_bert, batched=True)
        tsd = Dataset.from_pandas(pd.DataFrame({"text": x_test, "labels": y_test})).map(tf_bert, batched=True)
        vald_A = Dataset.from_pandas(pd.DataFrame({"text": x_val, "labels": y_val})).map(tf_bert, batched=True)

        roberta = bert_model(dataset, trd, tsd, "roberta-base")

        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        tf_bert = lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length")

        trd = Dataset.from_pandas(pd.DataFrame({"text": x_tr, "labels": y_tr})).map(tf_bert, batched=True)
        tsd = Dataset.from_pandas(pd.DataFrame({"text": x_test, "labels": y_test})).map(tf_bert, batched=True)
        vald_B = Dataset.from_pandas(pd.DataFrame({"text": x_val, "labels": y_val})).map(tf_bert, batched=True)

        codebert = bert_model(dataset, trd, tsd, "microsoft/codebert-base")
        print("MODELS FULL COMPARE")
        print("--------------------")
        print("Best Logistic Regression")
        print(eval_fun_br(test_tfidf, y_val, res_model))

        metrics = roberta.evaluate(vald_A)
        print(f"MODEL \"ROBERTA\" RESULTS")
        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Accuracy: {metrics['eval_accuracy']:.3f}")
        print(f"Precision: {metrics['eval_precision']:.3f}")
        print(f"Recall: {metrics['eval_recall']:.3f}")
        print(f"F1 Score: {metrics['eval_f1']:.3f}")
        print("--------------------------")

        metrics = codebert.evaluate(vald_B)
        print(f"MODEL \"CODEBERT\" RESULTS")
        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Accuracy: {metrics['eval_accuracy']:.3f}")
        print(f"Precision: {metrics['eval_precision']:.3f}")
        print(f"Recall: {metrics['eval_recall']:.3f}")
        print(f"F1 Score: {metrics['eval_f1']:.3f}")
        print("--------------------------")

    else:
        raise Exception("Invalid model name")


def model_classic_train(x_train, y_train):
    lr_clf = LogisticRegression()
    scoring = {"Accuracy": "accuracy", "F1_Score": "f1",
               "AUC": "roc_auc",
               "NegLogLoss": "neg_log_loss",
               "Precision": "precision",
               "Recall": "recall"
               }
    reg_cv = GridSearchCV(lr_clf, GSCV_parameters, cv=10, scoring=scoring, refit="F1_Score", verbose=0)
    reg_cv.fit(x_train, y_train)  # grid search CV makes division of dataset on test&train by itself

    print("GridSearch mean results beetwere:")
    print("CROSS VALID FINISHED, RESULTS:")
    print("Best params: ", reg_cv.best_estimator_.get_params())

    print("Best F1 Score result: ", reg_cv.best_score_)
    print("Index of the best model is: ", np.argmax(np.nan_to_num(reg_cv.cv_results_["mean_test_F1_Score"])))
    print("Mean within K10Folds validations F1 Score result: ",
          np.max(np.nan_to_num(reg_cv.cv_results_["mean_test_F1_Score"])))
    print("Mean within K10Folds validations Accuracy result: ",
          np.max(np.nan_to_num(reg_cv.cv_results_["mean_test_Accuracy"])))
    print("Mean within K10Folds validations AUC result: ", np.max(np.nan_to_num(reg_cv.cv_results_["mean_test_AUC"])))
    print("Mean within K10Folds validations NegLogLoss result: ",
          np.max(np.nan_to_num(reg_cv.cv_results_["mean_test_NegLogLoss"])))
    print("Mean within K10Folds validations Precision result: ",
          np.max(np.nan_to_num(reg_cv.cv_results_["mean_test_Precision"])))
    print("Mean within K10Folds validations Recall result: ",
          np.max(np.nan_to_num(reg_cv.cv_results_["mean_test_Recall"])))
    return reg_cv


def eval_fun_br(x_test, y_test, classifier):
    y_labels = np.array(classifier.predict(x_test))
    conf_matrix = confusion_matrix(y_test.to_numpy(), y_labels)
    f1 = f1_score(y_test, y_labels)
    precision = precision_score(y_test, y_labels)
    recall = recall_score(y_test, y_labels
                          )
    print("PREC: ", precision)
    print("REC: ", recall)
    print("F1 Score: ", f1)

    return (pandas.DataFrame({"Accuracy": np.round(np.mean([accuracy_score(y_test.to_numpy(),
                                                                           y_labels) for i in range(6)]), 3),
                              "F1": np.round(f1, 3),
                              "Precision": np.round(precision, 3),
                              "Recall": np.round(recall, 3),
                              "AUC": np.round(np.mean([roc_auc_score(y_test.to_numpy(),
                                                                     y_labels) for i in range(6)]), 3),
                              "Log loss": np.round(np.mean([log_loss(y_test.to_numpy(),
                                                                     y_labels) for i in range(6)]), 3)}, index=[0]),
            conf_matrix)


def bert_compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )

    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def bert_model(dataset, trd, tsd, model):
    model_impl = RobertaForSequenceClassification.from_pretrained(model, num_labels=2)

    targs = TrainingArguments(
        output_dir=f"./trained-{model}",
        eval_strategy="epoch",
        learning_rate=4e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model_impl,
        args=targs,
        train_dataset=trd,
        eval_dataset=tsd,
        compute_metrics=bert_compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    print(f"MODEL \"{model}\" RESULTS")
    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print(f"Accuracy: {metrics['eval_accuracy']:.3f}")
    print(f"Precision: {metrics['eval_precision']:.3f}")
    print(f"Recall: {metrics['eval_recall']:.3f}")
    print("--------------------------")
    print(f"F1 Score: {metrics['eval_f1']:.3f}")

    return trainer