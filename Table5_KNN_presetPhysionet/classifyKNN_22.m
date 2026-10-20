function results = classifyKNN_22(trainTable, testTable, varargin)
    % classifyKNN
    % KNN classification with k-fold cross-validation on training data,
    % then evaluation on test data.
    %
    % INPUTS:
    %   trainTable : feature table with FileName, Label, feature columns
    %   testTable  : feature table with FileName, Label, feature columns
    %
    % OPTIONAL NAME-VALUE INPUTS:
    %   'KFold'        : number of CV folds                                     (default: 10)
    %   'NumNeighbors' : number of nearest neighbours                           (default: 5)
    %   'Distance'     : 'euclidean','cosine','correlation','minkowski'         (default: 'cosine')
    %   'DistanceWeight': 'equal','inverse','squaredinverse'                    (default: 'equal')
    %   'Standardize'  : true/false                                             (default: false)
    %   'UnderSamplingFactor': [1, 2, 3]                                        (default: 2)
    %
    % OUTPUT:
    %   results : struct with CV and test metrics
    %
    % EXAMPLE:
    %   results = classifyKNN_22(trainData, testData);
    %   results = classifyKNN(trainTable, testTable, 'NumNeighbors', 3, 'Distance', 'cosine');

    %% ---- Parse inputs ----
    p = inputParser;
    addParameter(p, 'KFold',                10);
    addParameter(p, 'NumNeighbors',          5);
    addParameter(p, 'Distance',       'cosine');
    addParameter(p, 'DistanceWeight',  'equal');
    addParameter(p, 'Standardize',       false);
    addParameter(p, 'UnderSamplingFactor',   2);
    parse(p, varargin{:});
    opt = p.Results;

    %% ---- Prepare data ----
    [X_train, y_train, X_test, y_test, classNames] = prepareData_22(trainTable, testTable);

    fprintf('\n=== KNN | K: %d | Distance: %s | Folds: %d ===\n', ...
            opt.NumNeighbors, opt.Distance, opt.KFold);

    %% ---- K-Fold Cross-Validation on training data ----
    rng(42);
    cv      = cvpartition(y_train, 'KFold', opt.KFold, 'Stratify', true);
    
     cvMetricsAll = repmat(struct( ...
    'confMat', [], 'balancedAcc', [], 'accuracy', [], 'sensitivity', [], 'specificity', [], ...
    'precision', [], 'f1', [], 'mcc', [], 'kappa', [], ...
    'auc', [], 'totalInferenceTime_s', [], 'avgInferenceTime_ms', []), opt.KFold, 1);

    for f = 1:opt.KFold
        % Split into training and validation folds
        Xtr = X_train(cv.training(f), :);
        ytr = y_train(cv.training(f));
        Xte = X_train(cv.test(f), :);
        yte = y_train(cv.test(f));

        % Data Balancing - I : Partial undersampling
        idx_train_normal = find(ytr == -1);
        idx_train_abnormal = find(ytr == 1);
        
        nNormal_train = length(idx_train_normal);
        nAbnormal_train = length(idx_train_abnormal);
        
        if nNormal_train >= nAbnormal_train
            % Normal is majority → undersample normal
            target_majority = round(nAbnormal_train * opt.UnderSamplingFactor);
            target_majority = min(target_majority, nNormal_train);
            idx_majority_sampled = idx_train_normal(randperm(nNormal_train, target_majority));
            idx_minority_sampled = idx_train_abnormal; % keep all abnormal
        else
            % Abnormal is majority → undersample abnormal
            target_majority = round(nNormal_train * opt.UnderSamplingFactor);
            target_majority = min(target_majority, nAbnormal_train);
            idx_majority_sampled = idx_train_abnormal(randperm(nAbnormal_train, target_majority));
            idx_minority_sampled = idx_train_normal; % keep all normal
        end
        
        % Combine
        idx_partial_balanced = [idx_majority_sampled; idx_minority_sampled];
        idx_partial_balanced = idx_partial_balanced(randperm(length(idx_partial_balanced)));
        
        
        Xtr = Xtr(idx_partial_balanced, :);
        ytr = ytr(idx_partial_balanced);
        
        mdl = fitcknn(Xtr, ytr, ...
                      'NumNeighbors',   opt.NumNeighbors, ...
                      'Distance',       opt.Distance, ...
                      'DistanceWeight', opt.DistanceWeight, ...
                      'ClassNames',     classNames);

        % Predict labels + decision scores
        tic;
        [yPredFold, scores] = predict(mdl, Xte);
        inferenceTimeSec = toc;

        % Compute metrics for this fold
        cvMetricsAll(f) = computeMetrics_10(yte, yPredFold, classNames, ...
                                         sprintf('Fold-%d', f), ...
                                         inferenceTimeSec, numel(yte), scores);
    end

    %% ---- Aggregate across folds ----
    % Collect vectors across folds
    balAcc  = [cvMetricsAll.balancedAcc];
    acc     = [cvMetricsAll.accuracy];
    se      = [cvMetricsAll.sensitivity];
    sp      = [cvMetricsAll.specificity];
    pr      = [cvMetricsAll.precision];
    f1      = [cvMetricsAll.f1];
    mcc     = [cvMetricsAll.mcc];
    kap     = [cvMetricsAll.kappa];
    auc     = [cvMetricsAll.auc];

    % Compute mean ± std
    cvSummary.mean_balancedAcc  = mean(balAcc);cvSummary.std_balancedAcc  = std(balAcc);
    cvSummary.mean_accuracy     = mean(acc);   cvSummary.std_accuracy     = std(acc);
    cvSummary.mean_sensitivity  = mean(se);    cvSummary.std_sensitivity  = std(se);
    cvSummary.mean_specificity  = mean(sp);    cvSummary.std_specificity  = std(sp);
    cvSummary.mean_precision    = mean(pr);    cvSummary.std_precision    = std(pr);
    cvSummary.mean_f1           = mean(f1);    cvSummary.std_f1           = std(f1);
    cvSummary.mean_mcc          = mean(mcc);   cvSummary.std_mcc          = std(mcc);
    cvSummary.mean_kappa        = mean(kap);   cvSummary.std_kappa        = std(kap);
    cvSummary.mean_auc          = mean(auc);   cvSummary.std_auc          = std(auc);

    % Print summary
    fprintf('\n--- Mean ± Std across %d folds ---\n', opt.KFold);
    fprintf('BalancedAcc:  %.4f ± %.4f\n', cvSummary.mean_balancedAcc, cvSummary.std_balancedAcc);
    fprintf('Accuracy:     %.4f ± %.4f\n', cvSummary.mean_accuracy, cvSummary.std_accuracy);
    fprintf('Sensitivity:  %.4f ± %.4f\n', cvSummary.mean_sensitivity, cvSummary.std_sensitivity);
    fprintf('Specificity:  %.4f ± %.4f\n', cvSummary.mean_specificity, cvSummary.std_specificity);
    fprintf('Precision:    %.4f ± %.4f\n', cvSummary.mean_precision, cvSummary.std_precision);
    fprintf('F1-Score:     %.4f ± %.4f\n', cvSummary.mean_f1, cvSummary.std_f1);
    fprintf('MCC:          %.4f ± %.4f\n', cvSummary.mean_mcc, cvSummary.std_mcc);
    fprintf('Kappa:        %.4f ± %.4f\n', cvSummary.mean_kappa, cvSummary.std_kappa);
    fprintf('AUC:          %.4f ± %.4f\n', cvSummary.mean_auc, cvSummary.std_auc);

    %% ---- Train on full training set, evaluate on test set ----
    if opt.Standardize
        mu  = mean(X_train); sg = std(X_train);
        sg(sg == 0) = 1;
        X_train_sc = (X_train - mu) ./ sg;
        X_test_sc  = (X_test  - mu) ./ sg;
    else
        X_train_sc = X_train;
        X_test_sc  = X_test;
    end

    fullMdl = fitcknn(X_train_sc, y_train, ...
                      'NumNeighbors',   opt.NumNeighbors, ...
                      'Distance',       opt.Distance, ...
                      'DistanceWeight', opt.DistanceWeight, ...
                      'ClassNames',     classNames);

    tic;
    [testPreds, testScores] = predict(fullMdl, X_test_sc);
    inferenceTimeSec_test = toc;

    testMetrics = computeMetrics_11(y_test, testPreds, classNames, 'Test', ...
                    inferenceTimeSec_test, numel(y_test), testScores);

    %% ---- Package results ----
    results.classifier     = 'KNN';
    results.numNeighbors   = opt.NumNeighbors;
    results.distance       = opt.Distance;
    results.distanceWeight = opt.DistanceWeight;
    results.kFold          = opt.KFold;
    results.undersamplingFactor = opt.UnderSamplingFactor;
    results.cv.folds            = cvMetricsAll;   % per-fold metrics
    results.cv.summary          = cvSummary;      % mean ± std across folds

    % Default test evaluation
    results.test           = testMetrics;    % metrics at default threshold
    results.test.scores    = testScores;     % raw decision scores
    results.test.yTrue     = y_test;         % ground truth labels
    results.test.yPred     = testPreds;      % predicted labels (default threshold)
    
    % Model + Metadata    
    results.model          = fullMdl;
    results.classNames     = classNames;
end
