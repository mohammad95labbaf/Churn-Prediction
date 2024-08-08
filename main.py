# main.py
from config import Config
from dataset import load_dataset, preprocess_data, split_data
from classification import define_classifiers, evaluate_classifier, print_classification_results
from plot_confusion_matrix import plot_confusion_matrices
import argparse

"""
Main entry point for the project.

This script loads the dataset, preprocesses the data, splits the data into training and testing sets,
and evaluates the performance of various classifiers.
"""


## Just for Commitment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classifier', type=str, default='AdaBoost',
                        help='classifier to use. options: DecisionTree, KNN, SVM, Voting, RandomForest, Bagging, AdaBoost')
    parser.add_argument('-t', '--test-size', type=float, default=0.2,
                        help='test size for train_test_split. default: 0.2')
    parser.add_argument('-p', '--preprocessing-method', type=str, default='standardization',
                        help='preprocessing method to use. options: standardization, normalization, min-max-scaling, robust-scaling')
    args = parser.parse_args()


    valid_classifiers = ['DecisionTree', 'KNN', 'SVM', 'Voting', 'RandomForest', 'Bagging', 'AdaBoost']
    valid_preprocessing_methods = ['standardization', 'normalization', 'min-max-scaling', 'robust-scaling']

    if args.classifier not in valid_classifiers:
        print(f"Invalid classifier '{args.classifier}'. Please choose from: {', '.join(valid_classifiers)}")
        return

    if args.preprocessing_method not in valid_preprocessing_methods:
        print(f"Invalid preprocessing method '{args.preprocessing_method}'. Please choose from: {', '.join(valid_preprocessing_methods)}")
        return


    config = Config('config.yaml')
    data_settings = config.get_data_settings()
    model_settings = config.get_model_settings()

    df = load_dataset(data_settings['file_path'])
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, model_settings['random_state'], args.preprocessing_method)

    classifiers = define_classifiers(model_settings['random_state'])

    # Check if the chosen classifier is valid
    if args.classifier not in classifiers:
        print("Invalid classifier. Please choose from the following options:")
        print(", ".join(classifiers.keys()))
        return

    # Evaluate the chosen classifier
    clf = classifiers[args.classifier]
    accuracy, cm, cr = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
    print_classification_results(args.classifier, accuracy, cm, cr)
    plot_confusion_matrices(args.classifier, accuracy, cm, cr)

if __name__ == "__main__":
    main()

# Example Usage
# python main.py -c KNN, -t 0.5, -p robust-scaling
