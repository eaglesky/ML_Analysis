function [label_vector_train, instance_matrix_train, label_vector_test,...
instance_matrix_test] = split_data(label_vector, instance_matrix,...
num_test_per_class)

    [ncount, xout] = hist(label_vector, unique(label_vector));
    class_num = size(xout, 1);
    class_count = zeros(class_num, 1);
    test_ids = [];

    for i = 1:size(label_vector, 1)
        cur_label = label_vector(i);
        id = find(xout == cur_label);

        if class_count(id) < num_test_per_class
            class_count(id) = class_count(id) + 1;
            test_ids = [test_ids, i];
        end
    end

    label_vector_test = label_vector(test_ids, :);
    instance_matrix_test = instance_matrix(test_ids, :);

    label_vector_train = label_vector;
    instance_matrix_train = instance_matrix;
    label_vector_train(test_ids, :) = [];
    instance_matrix_train(test_ids, :) = [];


end
