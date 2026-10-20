function T = extractShenFeatures_11(presetHSData)
% Extracts full 137 Shen et al. features from presetHSData struct
%
% INPUT:
%   presetHSData : struct array from PresetDurationModule
% OUTPUT:
%   T            : table of features saved to shenFeatures

    N         = numel(presetHSData);
    results   = zeros(N, 137);
    fileNames = strings(N, 1);
    labels    = zeros(N, 1);

    fprintf('✔ Processing %d preset heart sounds\n', N);

    for k = 1:N
        fprintf('Processing %d / %d files\n', k, N);

        x         = presetHSData(k).signal;
        fs        = presetHSData(k).fs;
        fileNames(k) = presetHSData(k).filename;

        if isfield(presetHSData, 'label')
            lv = presetHSData(k).label;
            if isnumeric(lv); labels(k) = string(num2str(lv));
            else;             labels(k) = string(lv); end
        end

        %% ---- REFF (1–4) ----
        [pxx, f] = pwelch(x, hamming(1024), 512, 2048, fs);
        mask = (f >= 20 & f <= 800);
        pxx = pxx(mask); f = f(mask);

        intensity  = rms(x);
        centroid   = sum(f .* pxx) / sum(pxx);
        bandwidth  = sqrt(sum(((f - centroid).^2) .* pxx) / sum(pxx));
        [~, i]     = max(pxx);
        pitch      = f(i);
        simplicity = max(pxx) / sum(pxx);

        %% ---- STAT (5–8) ----
        kurt  = kurtosis(x);
        skew  = skewness(x);
        crest = peak2rms(x);
        var_x = var(x);

        %% ---- CWT–SVD (9–48) ----
        [cfs, ~] = cwt(x, fs, 'amor');
        C = abs(cfs);
        [U, R, V] = svd(C, 'econ');
        singVals = diag(R); singVals = singVals(1:5).';
        HU=zeros(1,5); GU=zeros(1,5);
        HV=zeros(1,5); GV=zeros(1,5);
        HQ=zeros(1,5); GQ=zeros(1,5);
        for i = 1:5
            HU(i) = shannonEntropy(U(:,i)); GU(i) = giniIndex(U(:,i));
            HV(i) = shannonEntropy(V(:,i)); GV(i) = giniIndex(V(:,i));
            HQ(i) = shannonEntropy(V(:,i)); GQ(i) = giniIndex(V(:,i));
        end
        Rdiag    = abs(singVals);
        CWT_feat = [singVals, HU, GU, HV, GV, Rdiag, HQ, GQ];
        assert(numel(CWT_feat) == 40, 'CWT feature count error');

        %% ---- MFCC–SVD (49–89) ----
        mf = mfcc(x, fs, 'NumCoeffs', 12, ...
                  'Window', hamming(256, 'periodic'), ...
                  'OverlapLength', round(0.015 * fs));
        mf = mf(:, 2:end);
        [U, R, V] = svd(mf, 'econ');
        singVals  = diag(R); singVals = singVals(1:5).';
        HU=zeros(1,5); GU=zeros(1,5);
        HV=zeros(1,5); GV=zeros(1,5);
        HQ=zeros(1,5); GQ=zeros(1,5);
        for i = 1:5
            HU(i) = shannonEntropy(U(:,i)); GU(i) = giniIndex(U(:,i));
            HV(i) = shannonEntropy(V(:,i)); GV(i) = giniIndex(V(:,i));
            HQ(i) = shannonEntropy(V(:,i)); GQ(i) = giniIndex(V(:,i));
        end
        Rdiag     = abs(singVals);
        loudness  = mean(sum(mf.^2, 2));
        MFCC_feat = [singVals, HU, GU, HV, GV, Rdiag, HQ, GQ, loudness];
        assert(numel(MFCC_feat) == 41, 'MFCC feature count error');

        %% ---- GMM (90–129) ----
        gm = fitgmdist(mf, 2, 'CovarianceType', 'full', ...
                       'RegularizationValue', 1e-5, ...
                       'Options', statset('MaxIter', 500));
        Dkeep = 10;
        meanFeat = []; covEigFeat = [];
        for c = 1:2
            mu       = gm.mu(c, 1:Dkeep);
            meanFeat = [meanFeat, mu]; %#ok<AGROW>
            Sigma    = gm.Sigma(1:Dkeep, 1:Dkeep, c);
            eigVals  = sort(eig(Sigma), 'descend');
            covEigFeat = [covEigFeat, eigVals(1:Dkeep)']; %#ok<AGROW>
        end
        GMM_feat = [meanFeat, covEigFeat];
        assert(numel(GMM_feat) == 40, 'GMM feature count error');

        %% ---- RQA (130–137) ----
        x_rqa = resample(x, 1, 10);
        m = 3; tau = 10;
        N_rqa = length(x_rqa) - (m - 1) * tau;

        if N_rqa < 100
            RQA_feat = zeros(1, 8);
        else
            X = zeros(N_rqa, m);
            for ii = 1:m
                X(:, ii) = x_rqa((1:N_rqa) + (ii-1)*tau);
            end
            D   = pdist2(X, X, 'euclidean');
            epsR = 0.1 * std(D(:));
            RP  = D <= epsR;
            RR  = mean(RP(:));
            RP_no_diag = RP; RP_no_diag(eye(size(RP))==1) = 0;
            diagLengths = [];
            for t = -(N_rqa-1):(N_rqa-1)
                d   = diag(RP_no_diag, t);
                lbl = bwlabel(d);
                for l = 1:max(lbl)
                    len = sum(lbl == l);
                    if len >= 2; diagLengths(end+1) = len; end %#ok<AGROW>
                end
            end
            if isempty(diagLengths); DET=0; L=0; Lmax=0;
            else
                DET=sum(diagLengths)/sum(RP(:));
                L=mean(diagLengths); Lmax=max(diagLengths);
            end
            vertLengths = [];
            for col = 1:N_rqa
                v   = RP(:, col); lbl = bwlabel(v);
                for l = 1:max(lbl)
                    len = sum(lbl == l);
                    if len >= 2; vertLengths(end+1) = len; end %#ok<AGROW>
                end
            end
            if isempty(vertLengths); LAM=0; TT=0; Vmin=0;
            else
                LAM=sum(vertLengths)/sum(RP(:));
                TT=mean(vertLengths); Vmin=max(vertLengths);
            end
            Ra = DET / (RR + eps);
            RQA_feat = [RR, DET, Ra, LAM, L, TT, Lmax, Vmin];
        end
        assert(numel(RQA_feat) == 8, 'RQA feature count error');

        %% ---- Combine ----
        results(k, :) = [intensity, bandwidth, pitch, simplicity, ...
                         kurt, skew, crest, var_x, ...
                         CWT_feat, MFCC_feat, GMM_feat, RQA_feat];
    end

    %% ---- Build and save table ----
    validIdx        = fileNames ~= "";
    results_valid   = results(validIdx, :);
    fileNames_valid = fileNames(validIdx);
    labels_valid    = labels(validIdx);

    fprintf('Keeping %d valid samples out of %d\n', sum(validIdx), N);

    baseNames = {'Intensity','Bandwidth','Pitch','Simplicity', ...
             'Kurtosis','Skewness','Crest','Variance'};
    CWT_names  = strcat("CWT_", string(1:40));
    MFCC_names = strcat("MFCC_", string(1:41));
    GMM_names  = strcat("GMM_", string(1:40));
    RQA_names = {'RR','DET','Ra','LAM','L','TT','Lmax','Vmin'};
    featureNames = [baseNames, CWT_names, MFCC_names, GMM_names, RQA_names];

    T = array2table(results_valid, 'VariableNames', featureNames);
    T = addvars(T, fileNames_valid, 'Before', 1, 'NewVariableNames', 'filename');
    T = addvars(T, labels_valid, 'After', width(T), 'NewVariableNames', 'label');


end

%% ---- Helpers ----
function H = shannonEntropy(x)
    x = abs(x(:)); x = x / (sum(x) + eps);
    H = -sum(x .* log(x + eps));
end

function G = giniIndex(x)
    x = abs(x(:)); x = sort(x); n = numel(x);
    G = (2 * sum((1:n)' .* x) / (n * sum(x))) - (n+1)/n;
end
