import time
import os
import tensorflow as tf
from src.models.neural_net import NeuralNet
from src.models.logistic_regression import LogisticRegression
from src.models.boosted_trees import BoostedTrees
from src.models.random_forest import RandomForest
from src.DataManager import DataManager
from src.utils import print_run_time, get_class_probabilities

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    csv_file = 'preprocessedData.csv'
    batch_size = 160  # TRAINING + TEST + VALIDATION BATCH SIZE

    manager = DataManager(csv_file, batch_size)

    #class_obj = NeuralNet(manager)
    #class_obj = LogisticRegression(manager)
    class_obj = BoostedTrees()
    #class_obj = RandomForest()

    model = class_obj()
    class_obj.save_model_diagram(model)

    # Load dataset using Data Manager's load_dataset function
    dataset = manager.load_dataset(csv_file, class_obj.name)

    # Split Data into Train, Test, Validation Sets
    train, validation, test = manager.get_dataset_partitions_tf(dataset)

    # Prefetch and Cache Data to help speed up training and evaluation
    train = train.cache().prefetch(tf.data.AUTOTUNE)
    validation = validation.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)
    

    # Train model using multiple epochs, with the validation
    # set being evaluated after each epoch to check for overfitting
    train_start = time.time()
    model.fit(
        train,
        epochs=class_obj.epochs,
        validation_data=validation
    )

    train_end = time.time()

    model.summary()

    # Evaluate model using test data that has been held out
    test_start = time.time()
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test, verbose=1)
    test_end = time.time()

    print("\n Test Loss: %.3f\n Test Accuracy: %.3f\n Test Precision: %.3f\n Test Recall: %.3f\n" % (test_loss, test_accuracy, test_precision, test_recall))
    print("Training Run Time: " + print_run_time(train_end - train_start))
    print("Evaluation Run Time: " + print_run_time(test_end - test_start))
    print("Total Run Time: " + print_run_time(test_end - train_start))


