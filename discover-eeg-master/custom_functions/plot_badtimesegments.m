function plot_badtimesegments(params,EEG,single_path)
% ,total_recording_length
% Recording ids
s_ids = cellfun(@(x) regexp(x,'.*(?=_eeg.set)','match','lineanchors'),{EEG.filename});

etc = {EEG.etc};
srates = {EEG.srate};
shapesrates  = size(srates );
disp(['srates size：', num2str(shapesrates )]);
disp(['srates ：', srates(1,1)]);
% Plot in batches of x recordings

x = 30;
nRec = length(EEG);
nBatches = ceil(nRec/x);
clear EEG;
total_preserved_length=0;
for iBatch=1:nBatches
    
    % Subset of recordings for this batch
    if nRec > x*iBatch
        bmask = (x*(iBatch-1)+1):x*iBatch;
    else
        bmask = (x*(iBatch-1)+1):nRec;
    end
    
    % Recording ids for this batch
    % For a short version of the recordings IDs: delete fields that are all the
    % same (e.g. if all sessions are ses-1)
    splitted_ids = cellfun(@(x) strsplit(x,'_'),s_ids(bmask),'UniformOutput',false);
    splitted_ids = vertcat(splitted_ids{:});
    mask = ones(1,size(splitted_ids,2));
    for i=1:size(splitted_ids,2)
        if (numel(unique(splitted_ids(:,i)))==1 && size(splitted_ids,1)>1)
            mask(i)=0;
        end
    end
    ids = splitted_ids(:,find(mask));
    ids = join(ids,'_',2);
    ids = insertBefore(ids,'_','\'); % Escape the underscores
        
    % Lengths of each recording
    betc = etc(bmask);        
    lengths = cell2mat(cellfun(@(x) length(x.clean_sample_mask), betc, 'UniformOutput',0)); 
    bsrates = cell2mat(srates(bmask));
    total_recording_length=lengths ./(bsrates*60);

    badsegs_batch = cell(1,length(bmask));
    num_bad_samples=[];
    for iRec = 1:length(bmask)
        mask = betc{iRec}.clean_sample_mask;
        num_bad_samples(iRec) = sum(mask == 0);  
        % Find transitions 0 to 1 and 1 to 0
        boundaries = find(diff([false ~mask(:)' false]));
        % Add first sample and last sample
        boundaries = [1, boundaries, lengths(iRec)];
        % Substract subsequent boundaries to get the duration of bad time
        % periods
        badsegs_iRec = boundaries(2:end) - boundaries(1:end-1);
        % Store bad timeperiods of this recording in a cell array
        badsegs_batch{iRec} = badsegs_iRec;
      
    end
    total_bad_length = num_bad_samples./(bsrates*60);
    %%
    % Find the maximum number of bad segments per recording in this batch
    % and create a matrix with so many rows
    nbadsegs = cellfun(@(x) length(x), badsegs_batch);
    segs = zeros(length(bmask),max(nbadsegs));
    for iRec = 1:length(bmask)
        segs(iRec,1:nbadsegs(iRec)) = badsegs_batch{iRec};
    end
    % Divide by sampling rate
    bsrates = cell2mat(srates(bmask))';
    segs = segs./(bsrates*60); % Divide by sampling rate
    %%
    % Calculate total bad time segments length per recording
%     total_bad_length = cellfun(@(x) sum(x), badsegs_batch, 'UniformOutput', false);
%     total_bad_length = cell2mat(total_bad_length'); % Convert to a matrix
%     shape2 = size(total_bad_length);
%     disp(['total_bad_length的形状为：', num2str(shape2)]);
%     % Divide by recording length to get percentage
%     total_recording_length = lengths./(cell2mat(srates(bmask))'*60);
%     shape0 = size(total_recording_length);
%     disp(['total_recording_length 的形状为：', num2str(shape0)]);
%     rejected_trials_percentage = (total_bad_length ./ total_recording_length) * 100;
%     shape1 = size(rejected_trials_percentage);
%     disp(['rejected_trials_percentage 的形状为：', num2str(shape1)]);
%     % Plot the rejected percentage
%     f2 = figure('units','normalized','outerposition',[0 0 1 1]);
%     bar(rejected_trials_percentage);
%     xticks(1:length(bmask));
%     box off;
%     set(gca,'xticklabel',ids(1:length(bmask)),'xticklabelrotation',45)
%     ylabel('Percentage of rejected trials');
%     title('Percentage of rejected trials');
    %%
    % Calculate rejected trials percentage
    rejected_trials_percentage = (total_bad_length ./total_recording_length) * 100;
    f2 = figure('units','normalized','outerposition',[0 0 1 1]);
    bar(rejected_trials_percentage);
    xticks(1:length(bmask));
    box off;
    set(gca,'xticklabel',ids(1:length(bmask)),'xticklabelrotation',45, ...
        'FontSize', 16, 'FontWeight', 'bold'); 
    ylabel('Percentage of rejected trials', 'FontSize', 16, 'FontWeight', 'bold');
%     title('Percentage of rejected trials');
    for i = 1:length(bmask)
        text(i, rejected_trials_percentage(i), sprintf('%.2f', rejected_trials_percentage(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 14, 'FontWeight', 'bold');
    end
    if single_path==1
        main_folder_path =params.PreprocessedDataPath;
        PreprocessedSub09DataPath = fullfile(main_folder_path, 'preprocessing_visualization-Sub09');
        if ~exist(PreprocessedSub09DataPath, 'dir')
            mkdir(PreprocessedSub09DataPath);
        end
        saveas(f2,fullfile(PreprocessedSub09DataPath, ['BadSegmentsPercentage_' num2str(iBatch) '.svg']),'svg');
    else
        saveas(f2,fullfile(params.FiguresPreprocessingPath, ['BadSegmentsPercentage_' num2str(iBatch) '.svg']),'svg');
    end

    %%
    % Calculate the duration of the time period to be retained.
    preserved_length = total_recording_length - total_bad_length;
    total_preserved_length = total_preserved_length + sum(preserved_length);    
    f3 = figure('units','normalized','outerposition',[0 0 1 1]);
    bar(preserved_length); 
    xticks(1:length(bmask));
    box off;
    set(gca,'xticklabel',ids(1:length(bmask)),'xticklabelrotation',45, ...
        'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Preserved length (minutes)', 'FontSize', 12, 'FontWeight', 'bold'); 
    if single_path==1
        main_folder_path =params.PreprocessedDataPath;
        PreprocessedSub09DataPath = fullfile(main_folder_path, 'preprocessing_visualization-Sub09');
        if ~exist(PreprocessedSub09DataPath, 'dir')
            mkdir(PreprocessedSub09DataPath);
        end
        saveas(f3,fullfile(PreprocessedSub09DataPath, ['RemainingDuration_' num2str(iBatch) '.svg']),'svg');
    else
        saveas(f3,fullfile(params.FiguresPreprocessingPath, ['RemainingDuration_' num2str(iBatch) '.svg']),'svg');
    end
    %%
    % Bar plots
    f = figure('units','normalized','outerposition',[0 0 1 1]);
    h = bar(1:size(segs,1), segs,'stacked','EdgeColor','none');
    set(h,'FaceColor','Flat');   
    for iRec=1:length(bmask)
        for k = 1:find((segs(iRec,:)~=0),1,'last')
            if mod(k,2) % Deal with the case in which the recording starts with a bad segment
                h(k).CData(iRec,:) = [0, 0.4470, 0.7410];
            else
                h(k).CData(iRec,:) = [1,0,0];
            end
        end
    end
    yl = ylim;
    ylim([0,yl(2)]);
    xticks(1:length(bmask));
    box off;
    set(gca,'xticklabel',ids(1:length(bmask)),'xticklabelrotation',45,'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Length of the recording (minutes)', 'FontSize', 12, 'FontWeight', 'bold');
    legend({'Good','Bad'},'Location','northeast');
    if strcmp(params.RejectBadTimeSegments,"on")
        title('Rejected bad segments');
    else
        title('Detected bad segments (not rejected)');
    end
    if single_path==1
        main_folder_path =params.PreprocessedDataPath;
        PreprocessedSub09Path = fullfile(main_folder_path, 'preprocessing_visualization-Sub09');
        if ~exist(PreprocessedSub09DataPath, 'dir')
            mkdir(PreprocessedSub09DataPath);
        end
        saveas(f,fullfile(PreprocessedSub09DataPath, ['BadSegments_' num2str(iBatch) '.svg']),'svg');
        save(fullfile(PreprocessedSub09DataPath, ['BadSegments_' num2str(iBatch) '.mat']),'segs','ids');
    else
        saveas(f,fullfile(params.FiguresPreprocessingPath, ['BadSegments_' num2str(iBatch) '.svg']),'svg');
        save(fullfile(params.FiguresPreprocessingPath, ['BadSegments_' num2str(iBatch) '.mat']),'segs','ids');
    end
%     close(f);
end
disp(['total_preserved_length：', num2str(total_preserved_length)]);
end

