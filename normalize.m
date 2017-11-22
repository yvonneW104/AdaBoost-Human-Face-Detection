function I = normalize(I)
    [N,~,~] = size(I);
    for i = 1:N
        image = I(i,:,:);
        sigma = std(image(:));
        I(i,:,:) = I(i,:,:) / sigma;
    end
end