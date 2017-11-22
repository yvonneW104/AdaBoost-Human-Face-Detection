function features = compute_features_test(II, filters)
    features = zeros(size(filters, 1), 1);
    for j = 1:size(filters, 1)
        [rects1, rects2] = filters{j,:};
        features(j,:) = apply_filter_test(II, rects1, rects2);
    end
end