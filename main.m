close all;
clear;

[label_vector, instance_matrix] = libsvmread('data/vehicle.scale');
[label_vector_train, instance_matrix_train] = ...
libsvmread('data/vehicle_train.scale');
[label_vector_test, instance_matrix_test] = ...
libsvmread('data/vehicle_test.scale');

[ndata, nfeature] = size(instance_matrix);

% Split original data into training data and testing data
[ncount, xout] = hist(label_vector, unique(label_vector));
num_test_per_class = floor(min(ncount)/4);

%[label_vector_train, instance_matrix_train, label_vector_test,...
%instance_matrix_test] = split_data(label_vector, instance_matrix,...
%num_test_per_class);

%libsvmwrite('data/vehicle_train.scale', label_vector_train,...
%instance_matrix_train);
%libsvmwrite('data/vehicle_test.scale', label_vector_test,...
%instance_matrix_test);

% Draw class distribution graph
figure;
[ncount_train, xout_train] = hist(label_vector_train, unique(label_vector_train));
bar(xout_train, ncount_train);
text(xout_train,ncount_train',num2str(ncount_train'),... 
'HorizontalAlignment','center',... 
'VerticalAlignment','bottom');
title('Class Distribution');

% Model selection using 10-fold cross validation

model_sel = [0, 0 , 0, 0];

for i = 1 : size(model_sel, 2)

    if i == 1 % SVM linear kernel, C
        LB = [1];
        UB = [50];
        x0 = 1;
    elseif i == 2 % SVM polynomial kernel, C, d, g, r
        LB = [1, 1, 0.001, -100];
        UB = [50, 10, 100, 100 ];
        x0 = [1, 1, 0.001, 0];
    elseif i == 3 % SVM RBF kernel, C, g
        LB = [1, 0.01];
        UB = [9000, 10];
        x0 = [1, 1];
    elseif i == 4 % SVM sigmoid kernel, C, g, r
        LB = [1, 0.01];
        x0 = [1, 0.01, 0];
    end

    if model_sel(i) == 1
        opts = psoptimset('PlotFcns',{@psplotbestf});
        [best_x, min_err] = patternsearch(@(x) err_cv(x, i, label_vector_train, ...
        instance_matrix_train), x0, [], [], [], [], LB, UB, [], opts);

        fprintf('Best paramters: ');
        disp(best_x);
        disp(['Optimized accuracy = ', num2str(100-min_err), ' %']);
    end

end

% Test on testing data set and draw confusion matrices
for i = 1:4
     if i == 0
         svm_params = '';
         id = 0;
         type = 'linear discriminant function';
     elseif i == 1
         id = 1;
         svm_params = '-s 0 -t 0 -c 44 -q';
         type = 'SVM classifier using linear kernel' ;
     elseif i == 2
         id = 1;
         svm_params = '-s 0 -t 1 -c 32 -d 4 -g 32 -r 96 -q';
         type = 'SVM classifier using polynomial kernel';
     elseif i == 3
         id = 1;
         svm_params = '-s 0 -t 2 -c 8192 -g 0.03125 -q';
         type = 'SVM classifier using RBF kernel';
     else
         id = 1;
         svm_params = '-s 0 -t 3 -c 808 -g 0.02 -r -0.3594 -q';
         type = 'SVM classifier using sigmoid kernel';
     end

    
    if id == 1
        
         svm_model = svmtrain(label_vector_train, instance_matrix_train, svm_params);
         predicted_labels = svmpredict(label_vector_test, instance_matrix_test, svm_model);

        [CM, order] = confusionmat(label_vector_test, predicted_labels);
        class_num = size(order, 1);
        class_names = [];
        for i = 1:class_num
            class_names = [class_names, num2str(order(i))];
        end

        figure;
        set(gcf,'Name',['Confusion Matrix of ', type]);
        draw_cm(CM, order, size(order, 1));
        xlabel('Prediction');
        ylabel('Truth');
    end
    fprintf('\n')
end


