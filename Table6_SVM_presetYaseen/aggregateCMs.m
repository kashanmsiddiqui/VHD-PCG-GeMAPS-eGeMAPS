% Number of folds
numFolds = numel(SVMresults.cv.folds);

% Initialize with zeros (same size as one confusion matrix)
aggConfMat = zeros(size(SVMresults.cv.folds(1).confMat));

% Loop through folds and sum
for k = 1:numFolds
    aggConfMat = aggConfMat + SVMresults.cv.folds(k).confMat;
end

% Display the aggregated confusion matrix
disp('Aggregated Confusion Matrix:');
disp(aggConfMat);
