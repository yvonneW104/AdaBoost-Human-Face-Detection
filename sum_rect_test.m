function result = sum_rect_test(II, offset, rect)
    x_off = offset(1);
    y_off = offset(2);

    x = rect(1);
    y = rect(2);
    w = rect(3);
    h = rect(4);

    a1 = II(y_off + y + h, x_off + x + w);
    a2 = II(y_off + y + h, x_off + x);
    a3 = II(y_off + y,     x_off + x + w);
    a4 = II(y_off + y,     x_off + x);

    result = a1 - a2 - a3 + a4;
end