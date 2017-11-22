function sum = apply_filter_test(II, rects1, rects2)
    sum = 0;
    % white rects
    for k = 1:size(rects1,1)
        r1 = rects1(k,:);
        w = r1(3);
        h = r1(4);
        sum = sum + sum_rect_test(II, [0, 0], r1) / (w * h * 255);
    end
    % black rects
    for k = 1:size(rects2,1)
        r2 = rects2(k,:);
        w = r2(3);
        h = r2(4);
        sum = sum - sum_rect_test(II, [0, 0], r2) / (w * h * 255);
    end
end