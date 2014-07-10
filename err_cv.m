function err = err_cv(X, classifier_id, labels, data)

    if classifier_id == 1 % SVM Linear kernel
        svm_params = ['-s 0 -t 0 -v 10 -q -c ', num2str(X)];
    elseif classifier_id == 2 % SVM polynomial kernel
        svm_params = ['-s 0 -t 1 -v 10 -q -c ', num2str(X(1)), ' -d ',...
        num2str(X(2)), ' -g ', num2str(X(3)), ' -r ', num2str(X(4))];
    elseif classifier_id == 3 % SVM RBF kernel
        svm_params = ['-s 0 -t 2 -v 10 -q -c ', num2str(X(1)), ' -g ', ...
        num2str(X(2))];
    elseif classifier_id == 4 % SVM sigmoid kernel
        svm_params = ['-s 0 -t 3 -v 10 -q -c ', num2str(X(1)), ' -g ', ...
        num2str(X(2)), ' -r ', num2str(X(3))];
    end

    acc = svmtrain(labels, data, svm_params);
    err = 100 - acc;
end
