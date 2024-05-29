function process_mat_data(ALLEEG,main_folder_path,PreprocessedDataPath)
%%   不用顾忌params，转epoch 和label mat格式
% clear all; close all;
% 创建一个空的 ALLEEG 结构
% ALLEEG = [];
% % 递归加载文件夹内所有 .set 文件
% ALLEEG = load_all_sets(main_folder_path, ALLEEG);
% 现在 ALLEEG 结构中包含了所有加载的 .set 文件的 EEG 数据
for iRec=1:length(ALLEEG)
  EEGtemp = eeg_checkset(ALLEEG(iRec),'loaddata');
  for i =1:length(EEGtemp.epoch)
    event_types = {EEGtemp.epoch(i).eventtype};
    if EEGtemp.event(i).epoch~=i
        EEGtemp.event(i).epoch=i;
    end           
  end 
    label= cellfun(@str2num,{EEGtemp.event.type});
    % 生成要保存的文件名
    id = regexp(ALLEEG(iRec).filename,'.*(?=(_eeg.set))','match');
    id = id{:};
    short_id = id(1:6);  % 提取 ID 的前 6 个字符
    % 创建以 short_id 为名的文件夹（如果不存在的话）
    folder_path = fullfile(PreprocessedDataPath, short_id);
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end
    newname = [id, '_', 'label'];
    filename1 = fullfile(folder_path,  [newname,'.mat']);
    save(filename1, 'label');    
    filename = [id,'.mat'];
    full_path = fullfile(folder_path, filename);
    save(full_path, 'EEGtemp');
end
end