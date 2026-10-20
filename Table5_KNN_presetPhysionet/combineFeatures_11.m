function T_merged = combineFeatures_11(shenT, gemapsT)

    % This function reads Shen and GeMPAS/eGemaps feature tables from 
    % workspace, and merges them such that 'filename' and 'label' remain 
    % same whereas features are concatenated.
    
    % --- Load both feature sets ---
    T1 = shenT;
    T2 = gemapsT;
    
    % --- Sanity check: ensure filename columns match type ---
    T1.filename = string(T1.filename);
    T2.filename = string(T2.filename);
    
    % --- Remove duplicate columns (filename + label) from second table ---
    T2_featuresOnly = T2(:, 3:end-1);   % drop first (filename), second (unknown) and last (label)
    
    % --- Concatenate feature columns ---
    T_merged = [T1(:, 1), T1(:, 2:end-1), T2_featuresOnly, T1(:, end)];
    
    % --- Rename columns for clarity ---
    % Collect feature names from both sets
    featureNames = [T1.Properties.VariableNames(2:end-1), ...
                    T2.Properties.VariableNames(3:end-1)];
    
    T_merged.Properties.VariableNames = [{'filename'}, featureNames, {'label'}];
    

end
