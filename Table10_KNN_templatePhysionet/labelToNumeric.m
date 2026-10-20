function y = labelToNumeric(labelCol)
    if isnumeric(labelCol)
        y = labelCol;
        return;
    end
    labelCol = string(labelCol);
    uniq = unique(labelCol);
    y = zeros(numel(labelCol), 1);
    
    y = double(labelCol == "1");  % '1' = 1, '-1' = 0
    
end