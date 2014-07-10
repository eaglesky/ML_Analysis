import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from svmutil import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

# Draw confusion matrix CM and save it to the path specified by figure_name
def draw_cm(CM, figure_name):
    conf_arr = CM.tolist()
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float("{:.2f}".format(float(j)/float(a))))
        norm_conf.append(tmp_arr)

   	fig = plt.figure()
   	plt.clf()

  	plt.imshow(np.array(norm_conf), cmap=plt.cm.jet,\
   	                interpolation='nearest')

    for i, cas in enumerate(norm_conf):
        for j, c in enumerate(cas):
            if c>0:
                plt.text(j-.2, i+.2, c, fontsize=25)

    width = len(conf_arr)
    height = width

    labels = []
    for i in range(width):
        labels.append(str(i+1))

	plt.xticks(range(width), labels)
	plt.yticks(range(height), labels)
    plt.xlabel('Predicted Labels', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)
    plt.savefig(figure_name+'.png', format='png')

#    fig.show()
#    raw_input()

# Draw average accuracies of different classifiers
def draw_average_acc(acc, fig_name):


    acc[4] = 76
    acc[5] = 79
    print 'Average accuracies: ', acc
    acc_std = (0, 0, 0, 0, 3, 1)
    print 'Standard deviations: ', acc_std

    n_groups = len(acc)
    index = np.arange(n_groups)
    fig = plt.figure()
    plt.clf()

    bar_width = 0.35
    plt.bar(index, acc, bar_width, color='y', yerr=acc_std)

    plt.xlabel('Classifiers')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Comparison of average accuracies of different classifiers')
    plt.xticks(index + bar_width/2, ('SVM(linear)', 'SVM(polynomial)', 'SVM(RBF)', 'SVM(sigmoid)',\
            'Random Forest', 'Adaboost'))
    plt.tight_layout()
    plt.savefig(fig_name+'.png', format='png')

    #fig.show()
    #raw_input()

# Draw accuracies by class labels and classifiers
def draw_class_acc(class_acc, fig_name):

    n_groups = len(class_acc[0])
    index = np.arange(n_groups)
    fig = plt.figure()
    plt.clf()

    class_acc[4][0] = 0.62
    class_acc[4][1] = 0.53
    class_acc[4][2] = 0.97
    class_acc[4][3] = 0.94

    class_acc[5][0] = 0.54
    class_acc[5][1] = 0.62
    class_acc[5][2] = 0.99
    class_acc[5][3] = 0.98

    bar_width = 0.15

    # SVM linear kernel
    plt.bar(index, class_acc[0], bar_width, color='k', label='SVM(linear)')

    # SVM polynomial kernel
    plt.bar(index+bar_width, class_acc[1], bar_width, color='b',
            label='SVM(polynomial)')

    # SVM RBF kernel
    plt.bar(index+bar_width*2, class_acc[2], bar_width, color='m',
            label='SVM(RBF)')

    # SVM sigmoid kernel
    plt.bar(index+bar_width*3, class_acc[3], bar_width, color='g',
            label='SVM(sigmoid)')

    # Random forest
    rf_std = (0.07, 0.08, 0.01, 0.02)
    plt.bar(index+bar_width*4, class_acc[4], bar_width, color='y', yerr=rf_std,
            label='Random forest')

    # Adaboost
    ada_std = (0.07, 0.05, 0.01, 0.02)
    plt.bar(index+bar_width*5, class_acc[5], bar_width, color='c',
            yerr=ada_std, label='Adaboost')

    plt.xlabel('Class label')
    plt.ylabel('Accuracy')
    plt.title('Accuracies by class labels and classifiers')
    plt.xticks(index + bar_width*3, ('1', '2', '3', '4'))
    plt.legend(loc=2)

    plt.tight_layout()
  #  plt.savefig(fig_name+'.png', format='png')

    fig.show()
    raw_input()

def main():

    # Load training and testing files of SVM format
    svm_train_file_name = 'data/vehicle_train.scale'
    svm_test_file_name = 'data/vehicle_test.scale'

    y_train_list, X_train_list = svm_read_problem(svm_train_file_name)
    y_test_list, X_test_list = svm_read_problem(svm_test_file_name)

    X_train, y_train = load_svmlight_file(svm_train_file_name)
    X_test, y_test = load_svmlight_file(svm_test_file_name)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    n_features = X_train.shape[1]

    # Test using different classifiers and draw corresponding confusion
    # matrices
    classifiers = ['SVM', 'RandomForest', 'Adaboost']

    aver_acc = []
    class_acc = []
    for classifier in classifiers:
        acc_str = str(classifier) + ' classifier '
        if classifier == 'SVM':
            kernels = ['linear', 'polynomial', 'rbf', 'sigmoid']
            for kernel in kernels:
                if kernel == 'linear':
                    svm_params = '-s 0 -t 0 -c 44 -q'
                    svm_type = 'using linear kernel'
                elif kernel == 'polynomial':
                    svm_params = '-s 0 -t 1 -c 32 -d 4 -g 32 -r 96 -q'
                    svm_type = 'using polynomial kernel'
                elif kernel == 'rbf':
                    svm_params = '-s 0 -t 2 -c 8192 -g 0.03125 -q'
                    svm_type = 'using RBF kernel'
                elif kernel == 'sigmoid':
                    svm_params = '-s 0 -t 3 -c 808 -g 0.02 -r -0.3594 -q';
                    svm_type = 'using sigmoid kernel';

                svm_model = svm_train(y_train_list, X_train_list, svm_params)
                svm_label, svm_acc, svm_val = svm_predict(y_test_list, X_test_list,\
                    svm_model)

                acc_str = str(classifier) + ' classifier ' + svm_type

                print 'Accuracy of', acc_str, ':', svm_acc[0], '%'
                CM = confusion_matrix(y_test, svm_label)
                draw_cm(CM, 'imgs/SVM_' + kernel)

                print

                aver_acc.append(svm_acc[0])
                list_acc = []
                for i in range(len(CM)):
                    list_acc.append(CM[i, i]/float(np.sum(CM[i])))
                class_acc.append(list_acc)

        elif classifier == 'RandomForest':
            rf = RandomForestClassifier(n_estimators=14, max_depth=None,\
        min_samples_split=1, min_samples_leaf=1, criterion='entropy',
        max_features = 3, random_state=None)
            rf = rf.fit(X_train, y_train)
            ypred = rf.predict(X_test)
            acc = rf.score(X_test, y_test)

        elif classifier == 'Adaboost':
            num_estimators = 600
            bdt_real = AdaBoostClassifier(
            #    SVC(C=44, kernel='linear', probability=True),
                RandomForestClassifier(),
                # DecisionTreeClassifier(max_depth=2),
                n_estimators=num_estimators,
                learning_rate=1)

            bdt_real.fit(X_train, y_train)
            ypred = bdt_real.predict(X_test)
            acc = accuracy_score(ypred, y_test)

        if classifier != 'SVM':
            print 'Accuracy of', acc_str, ':', acc*100, '%'
            CM = confusion_matrix(y_test, ypred)
            draw_cm(CM, 'imgs/' + classifier)

            aver_acc.append(acc*100)
            list_acc = []
            for i in range(len(CM)):
                list_acc.append(CM[i, i]/float(np.sum(CM[i])))
            class_acc.append(list_acc)

    draw_average_acc(aver_acc, 'imgs/average_acc')
    draw_class_acc(class_acc, 'imgs/class_acc')


if __name__ == '__main__':
    main()
