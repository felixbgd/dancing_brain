%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Imaging the dancing brain" paper - Bigand et al. (2024) %
%                   ERP analysis (audio)                   %
%%%%%%%%% Félix Bigand, NPA Lab, 2022-2024 - Rome %%%%%%%%%%

%% Import libraries, initialise fieldtrip

clc
clear all
close all


addpath(genpath('.\functions\'));       % personal functions provided in github repo
addpath(genpath('.\mat\'));   % folder with EEG data (.mat), to be adapted

% toolboxes to download separately: fieldtrip, mTRF
addpath('.\fieldtrip-20221014\');
addpath(genpath(['.\fieldtrip-20221014\external\eeglab\']));
addpath('.\fieldtrip-20221014\external\brewermap\')     % for colormaps
% addpath(['.\mtrf\']); 
ft_defaults;


%% Set parameters, folders, paths etc

output_dir = '.\results_step3_ERP_audio\';
if ~isfolder(output_dir)  mkdir(output_dir);   end

NB_TRIALS=32;
NB_SONGS=8;
fps=100;

stim_names = {'song1','song2','song3','song4','song5','song6','song7','song8'};

% Compute length of songs/temporal structure
song_bpms = [111.03,116.07,118.23,118.95,120.46,125.93,128.27,129.06];
periodbeat = (60./song_bpms);
musParts_beats = [1,17,33,49,65,81];       % start drums , bass, harmony, voice, end
beats_tFrames = [];         musParts_tFrames_s = [];
for i=1:NB_SONGS
    beats_tFrames(i,:) = linspace(0,80*periodbeat(i),81); % Because 80 beats for each song
    musParts_tFrames_s(i,:) = beats_tFrames(i,musParts_beats);
end
musParts_tFrames = round(musParts_tFrames_s * fps);

%% Load data (already stored as .mat files)

disp('LOADING DATA...')
dir_inputEEG = '.\mat\cleaned_8Hz_100fps\';     % ADAPT WITH YOUR PATH
fileNames = {dir([dir_inputEEG '*.mat']).name};

load(['.\\songNums_allTrials_allSubj.mat']);
load(['.\\conds_allTrials_allSubj.mat']);

% Directories for predictor/stimuli to predict the EEG
% Audio waveforms --> music model
stim_style = 'all';
predictor_audio = 'StimAcoustics';
allpred_audio = load(['.\\' predictor_audio]);

% Audio Lag  (s)
audioLag = 0.0535;

% Loop for epoching EEG
iParticipant = 1;
for p = [1:40 43:44 47:50 55:56 59:80]          % only participants with OK mocap data
    disp(['Participant ' num2str(p)])
    
    if p < 10  numParticipant = ['0' num2str(p)];
    else numParticipant = num2str(p);
    end
    
    % Load EEG data of the participant
    load([dir_inputEEG fileNames{p}]);
    tr = data_bp_rs_asr_ica_interp_all.trialinfo;  statusevent = data_bp_rs_asr_ica_interp_all.statusevent;
    data_bp_rs_asr_ica_interp_all = ft_preprocessing([],data_bp_rs_asr_ica_interp_all);

    % Retrieve EEG scalp data
    data_bp_rs_asr_ICA_interp           = EEG_removeChannels(data_bp_rs_asr_ica_interp_all,{'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6'});

    %%% ERPs %%%
    %%% Parameters to epoch ERPs %%%
    win_start=-0.25;   win_stop=0.3;
    n_offsets_s = [win_start win_stop];
    data_ERP = data_bp_rs_asr_ICA_interp; 
    
    % Loop over conditions
    new_sampling=data_ERP.fsample;
    n_offsets = round(n_offsets_s * new_sampling);   
    for cond=1:4
        all_tr_same_cond=[];
        
        % Find trials corresponding to each condition
        tr_cond = find(tr==cond)';  
        % Exception message for one trial (the last one) of one dyad absent
        if ismember(p,[73,74]) && length(find(tr_cond==32))>0    
            cond_missing = cond;  tr_missing_in_cond=find(tr_cond==32);
            tr_cond(tr_missing_in_cond)=[]; 
        end
        
        % Loop over trials
        iTrial=1;
        for ii = tr_cond       % For each trial that you wanna analyse
            disp(['TRIAL ' num2str(ii)]);

            % Compute first onset frame
            prestim     = round((1+audioLag) * fps);   % Because there was 1s of prestim in our EEG first definition (prepro)
            idxStartTr  = (ii-1)*4 + 1;       % Because 4 triggers per trial (startTrial, startSong, stopSong, stopTrial)
            startSong_s = (statusevent(idxStartTr+1).sample-statusevent(idxStartTr).sample)/1024;         % first music onset (startSong) in s (there was silent/still period between startTrial and startSong)
            startSong   = round(startSong_s * fps);
            
            % Check what songs were played for both subjects
            if mod(p,2)==1 iOtherParticipant = p+1; end
            if mod(p,2)==0 iOtherParticipant = p-1; end
            songNum_self  = songNums_allTrials_allSubj(p,ii);
            songNum_other = songNums_allTrials_allSubj(iOtherParticipant,ii);
            
            lensong       = min(musParts_tFrames([songNum_self,songNum_other] , end));  % Trim stop to the common time they listened to music in the trial
            cutOnset      = 2*round(periodbeat(songNum_self) * fps);  % Trim start after 2 beats of music (to avoid confounding effect of first "startle" responses at beginning, see TRF litterature)

            %%% Find audio/music salient peaks %%%
            pred_idx = find(strcmp(allpred_audio.stim_names,['song' num2str(songNum_self) '_' stim_style '.wav']));
            Stim_one_trial_audio = [allpred_audio.specflux_avg{pred_idx}']; sz_feat=[1]; feat_names={'T specflux'};
            Stim_one_trial_audio = Stim_one_trial_audio(:,1:lensong);
            
            % Find peaks
            mean_data = mean(Stim_one_trial_audio); std_data = std(Stim_one_trial_audio);
            thresh = 3*std_data - mean_data;
            [peaks,locpeaks] = findpeaks(Stim_one_trial_audio,'MinPeakHeight',thresh);
            audio_peaks = zeros(1,length(Stim_one_trial_audio));
            audio_peaks(locpeaks) = peaks;

            Stim_one_trial_audio_peaks = audio_peaks;
            Stim_one_trial_audio_peaks(1:cutOnset)=0;

            % Define onset times of the epochs based on these peaks
            onset_times = find(Stim_one_trial_audio_peaks>0);

            %%% Epoch the EEG %%%
            if length(onset_times)>0
                cfg              = [];
                cfg.trials       = ii;
                EEG_of_one_trial = ft_selectdata(cfg,data_ERP);
                    
                note_onsets = onset_times + startSong + prestim + EEG_of_one_trial.sampleinfo(1);     % Add the timeframe of sampleinfo + EEG_of_one_trial.sampleinfo(1); 
                cfg = [];
                cfg.trl(:,1) = note_onsets + n_offsets(1);        % start
                cfg.trl(:,2) = note_onsets + n_offsets(2);        % stop
                cfg.trl(:,3) = n_offsets(1);                      % store start
                cfg.trl(:,4) = Stim_one_trial_audio_peaks(onset_times);     % store audio peak value
                
                % Cut / epoch
                EEG_epochs_of_one_trial  = ft_redefinetrial(cfg,EEG_of_one_trial);
                all_tr_same_cond{iTrial} = EEG_epochs_of_one_trial;

                iTrial=iTrial+1;
            end
        end
        
        % Create ft struct with epochs of this condition and participant
        all_erp_same_cond         = ft_appenddata([],all_tr_same_cond{:});
        all_erp_same_cond.fsample = new_sampling;

        % Store epochs for each participant and condition
        all_tr_all_cond{iParticipant,cond} = all_erp_same_cond;
    end
    iParticipant = iParticipant +1;
    figure;

    % Empty plot just to get a feedback on advancement of the analysis
    exportgraphics(gcf,[output_dir '\indiv' num2str(iParticipant) '.jpg']);
    close;
end


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  PLOT ERPS                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pool epochs that correspond to "high" vs. "low" peak values
for p=1:iParticipant-1
    for cond=1:4
        disp(['COND ' num2str(cond)])
        disp(['PARTICIPANT ' num2str(p)])

        % Retrieve epochs per cond per participant
        erp_cond = all_tr_all_cond{p,cond};

        % Find lowest peaks of participant
        Ngroups              = round(0.2*length(erp_cond.trialinfo(:,1)));
        [~,trialsLow]        = mink(erp_cond.trialinfo(:,1),Ngroups);
        [~,trialsHigh]       = maxk(erp_cond.trialinfo(:,1),Ngroups);
        values_low{cond,p}   = erp_cond.trialinfo(trialsLow);
        values_high{cond,p}  = erp_cond.trialinfo(trialsHigh);

        % Average epoch per cond per Low/High per participant
        cfg                         = [];
        cfg.preproc.demean          = 'yes';
        cfg.preproc.baselinewindow  = [-0.25 0];   % using the mean activity in this window
        cfg.trials                  = trialsLow;
        avg_ERP_part_low{p,cond}    = ft_timelockanalysis(cfg, erp_cond);

        cfg                         = [];
        cfg.preproc.demean          = 'yes';
        cfg.preproc.baselinewindow  = [-0.25 0];   % using the mean activity in this window
        cfg.trials                  = trialsHigh;
        avg_ERP_part_high{p,cond}   = ft_timelockanalysis(cfg, erp_cond);
    end
end

% Grand Average per condition
for cond=1:4   
    % Low
    cfg = [];
    cfg.keepindividual = 'yes';
    
    avg_ERP_part_cond_low     = {avg_ERP_part_low{:,cond}};
    avg_ERP_allPart_low{cond} = ft_timelockgrandaverage(cfg,avg_ERP_part_cond_low{:});  

    cfg = [];
    GA_ERP_low{cond}          = ft_timelockgrandaverage(cfg,avg_ERP_part_cond_low{:}); 
    GA_ERP_low{cond}.stderr   = sqrt(GA_ERP_low{cond}.var) / sqrt(GA_ERP_low{cond}.dof(1));

    % High
    cfg = [];
    cfg.keepindividual = 'yes';
    
    avg_ERP_part_cond_high     = {avg_ERP_part_high{:,cond}};
    avg_ERP_allPart_high{cond} = ft_timelockgrandaverage(cfg,avg_ERP_part_cond_high{:});  

    cfg = [];
    GA_ERP_high{cond}          = ft_timelockgrandaverage(cfg,avg_ERP_part_cond_high{:});  
    GA_ERP_high{cond}.stderr   = sqrt(GA_ERP_high{cond}.var) / sqrt(GA_ERP_high{cond}.dof(1));
end

% Find clusters showing "low" / "high" difference in ERP amplitude
% Cluster-based permutation
% At channel of interest given by TRF results --> Fz
cluster_timepoints=[];
cond_names={'YV-SM','YV-DM','NV-SM','NV-DM'};
for cond=1:4   
    cfg = [];
    cfg.channel          = {'Fz'};
    cfg.method           = 'ft_statistics_montecarlo';  % use the Monte Carlo method to calculate probabilities
    cfg.statistic        = 'ft_statfun_depsamplesT';    % use the dependent samples T-statistic as a measure to evaluate the effect at each sample
    cfg.correctm         = 'cluster';
    cfg.clusteralpha     = 0.05;                        % threshold for the sample-specific test, is used for thresholding
    cfg.clusterstatistic = 'maxsum';
    cfg.clusterthreshold = 'nonparametric_common';
    cfg.tail             = 0;                           % test the left, right or both tails of the distribution
    cfg.clustertail      = 0;
    cfg.alpha            = 0.05;                        % alpha level of the permutation test
    cfg.correcttail      = 'alpha';                     % see https://www.fieldtriptoolbox.org/faq/why_should_i_use_the_cfg.correcttail_option_when_using_statistics_montecarlo/
    cfg.computeprob      = 'yes';
    cfg.numrandomization = 1000;                         % number of random permutations
    cfg.neighbours       = [];        % the neighbours for each sensor to form clusters
    
    nsubj                = size(avg_ERP_part_low,1);
    design               = zeros(2,2*nsubj);
    design(1,:)          = [ones(1,nsubj) ones(1,nsubj)*2];
    design(2,:)          = [1:nsubj 1:nsubj];
    
    
    cfg.design   = design; % design matrix
    cfg.ivar     = 1;      % the 1st row codes the independent variable (sedation level)
    cfg.uvar     = 2;      % the 2nd row codes the unit of observation (subject)
    
    stat_cluster = ft_timelockstatistics(cfg, avg_ERP_allPart_high{cond}, avg_ERP_allPart_low{cond});
    
    % retrieve timeframes of clusters (and probs)
    if isfield(stat_cluster,'posclusterslabelmat') && isfield(stat_cluster,'negclusterslabelmat') && ~isempty(stat_cluster.posclusters) && ~isempty(stat_cluster.negclusters)
        pos = (stat_cluster.posclusterslabelmat~=0) & stat_cluster.mask; 
        neg = (stat_cluster.negclusterslabelmat~=0) & stat_cluster.mask;
        cluster_timepoints{cond} = pos | neg;

        pos_prob=[stat_cluster.posclusters.prob]; neg_prob=[stat_cluster.negclusters.prob];
        cluster_pos_prob{cond} = pos_prob(pos_prob<0.05);
        cluster_neg_prob{cond} = neg_prob(neg_prob<0.05);
    elseif isfield(stat_cluster,'posclusterslabelmat') && ~isempty(stat_cluster.posclusters)
        pos = (stat_cluster.posclusterslabelmat~=0) & stat_cluster.mask; 
        cluster_timepoints{cond} = pos;
        
        pos_prob=[stat_cluster.posclusters.prob];
        cluster_pos_prob{cond} = pos_prob(pos_prob<0.05);
    elseif isfield(stat_cluster,'negclusterslabelmat') && ~isempty(stat_cluster.negclusters)
        neg = (stat_cluster.negclusterslabelmat~=0) & stat_cluster.mask;
        cluster_timepoints{cond} = neg;
        
        neg_prob=[stat_cluster.negclusters.prob];
        cluster_neg_prob{cond} = neg_prob(neg_prob<0.05);
    else
        cluster_timepoints{cond}=[];
        cluster_pos_prob{cond}=[];
        cluster_neg_prob{cond}=[];
    end
    
end

% Save clusters
save([output_dir 'cluster_probs'],'cluster_pos_prob','cluster_neg_prob');

% Compute cluster topos for all identified clusters
max_store = [];     % to store the max of cluster topos across all conds for future plots
for cond = 1:4
    cfg=[];
    cfg.parameter='avg';
    cfg.operation='x1-x2'; 
    diff_GA{cond} = ft_math(cfg,GA_ERP_high{cond},GA_ERP_low{cond});
    

    time = GA_ERP_low{cond}.time;
    if ~isempty(cluster_timepoints{cond})
        start_stop_cluster = diff([0 cluster_timepoints{cond} 0]);
        if start_stop_cluster(end)==-1 && cluster_timepoints{cond}(end-1)==1         % there is a cluster that started earlier, and never stops
            start_stop_cluster(end-1)=-1; start_stop_cluster(end)=0;
        elseif start_stop_cluster(end)==-1 && cluster_timepoints{cond}(end-1)==0     % this cluster only starts on the frame of the end, don't count it
            start_stop_cluster(end-1)=0; start_stop_cluster(end)=0;
        end
        nb_areas = length(find(start_stop_cluster==1));
        if nb_areas > 0
            start_frames = find(start_stop_cluster==1);
            stop_frames = find(start_stop_cluster==-1);
            start_times = time(find(start_stop_cluster==1));
            stop_times = time(find(start_stop_cluster==-1));
            for area=1:nb_areas
                mean_topo = mean(diff_GA{cond}.avg(:,start_frames(area):stop_frames(area)),2);
                max_store(cond,area) = max(abs(mean_topo(:)));
            end
        end
    end
end

% Plot cluster topos
absMax = max(abs(max_store(:)));
for cond = 1:4
    cfg=[];
    cfg.parameter='avg';
    cfg.operation='x1-x2'; 
    diff_GA{cond} = ft_math(cfg,GA_ERP_high{cond},GA_ERP_low{cond});
    

    time = GA_ERP_low{cond}.time;
    if ~isempty(cluster_timepoints{cond})
        start_stop_cluster = diff([0 cluster_timepoints{cond} 0]);
        if start_stop_cluster(end)==-1 && cluster_timepoints{cond}(end-1)==1         % there is a cluster that started earlier, and never stops
            start_stop_cluster(end-1)=-1; start_stop_cluster(end)=0;
        elseif start_stop_cluster(end)==-1 && cluster_timepoints{cond}(end-1)==0     % this cluster only starts on the frame of the end, don't count it
            start_stop_cluster(end-1)=0; start_stop_cluster(end)=0;
        end
        nb_areas = length(find(start_stop_cluster==1));
        if nb_areas > 0
            start_frames = find(start_stop_cluster==1);
            stop_frames = find(start_stop_cluster==-1);
            start_times = time(find(start_stop_cluster==1));
            stop_times = time(find(start_stop_cluster==-1));
            for area=1:nb_areas
                cfg = [];
                cfg.xlim = [start_times(area) stop_times(area)];   % time interval of the subplot
                cfg.zlim = [-absMax absMax];
                cfg.comment='no';
                cfg.colormap = '*RdBu';
                cfg.colorbar = 'yes';
                cfg.layout      = 'biosemi64';
                cfg.interactive = 'no';
                ft_topoplotER(cfg, diff_GA{cond});
                exportgraphics(gcf,[output_dir '\' cond_names{cond} '_CLUSTER' num2str(area) '.jpg']);
                exportgraphics(gcf,[output_dir '\' cond_names{cond} '_CLUSTER' num2str(area) '.pdf']);
                close;
            end
        end
    end
end


% Plot ERPs "low" / "high" at electrode of interest
% + gray shaded areas at the timestamps of identified clusters
cond_label = {'YV-SM','YV-DM','NV-SM','NV-DM'};
maxAbsZ_low = max([max(abs(GA_ERP_low{1}.avg)) max(abs(GA_ERP_low{2}.avg)) max(abs(GA_ERP_low{3}.avg)) max(abs(GA_ERP_low{4}.avg))]);
maxAbsZ_high = max([max(abs(GA_ERP_high{1}.avg)) max(abs(GA_ERP_high{2}.avg)) max(abs(GA_ERP_high{3}.avg)) max(abs(GA_ERP_high{4}.avg))]);
maxAbsZ = max([maxAbsZ_low maxAbsZ_high]);
figure;
set(gcf, 'Position',  [0, 0, 1800, 1000])
cfg=[];
cfg.channel='Fz';
cfg.ylim = [-maxAbsZ maxAbsZ];
cfg.figure = 'gca';
for cond=1:4
    subplot(2,4,cond)
    time = GA_ERP_low{cond}.time;
    RFT_singleplotER(cfg,GA_ERP_low{cond},GA_ERP_high{cond});
    if ~isempty(cluster_timepoints{cond})
        start_stop_cluster = diff([0 cluster_timepoints{cond} 0]);
        if start_stop_cluster(end)==-1 && cluster_timepoints{cond}(end-1)==1         % there is a cluster that started earlier, and never stops
            start_stop_cluster(end-1)=-1; start_stop_cluster(end)=0;
        elseif start_stop_cluster(end)==-1 && cluster_timepoints{cond}(end-1)==0     % this cluster only starts on the frame of the end, don't count it
            start_stop_cluster(end-1)=0; start_stop_cluster(end)=0;
        end
        nb_areas = length(find(start_stop_cluster==1));
        if nb_areas > 0
            start_times = time(find(start_stop_cluster==1));
            stop_times = time(find(start_stop_cluster==-1));
            for area=1:nb_areas
                patch([start_times(area) stop_times(area) stop_times(area) start_times(area)], [-2*maxAbsZ -2*maxAbsZ 2*maxAbsZ 2*maxAbsZ], [0.3 0.3 0.3], 'FaceAlpha', 0.2, 'EdgeColor', 'none' )
            end
        end
    end
    xticks(win_start:0.05:win_stop); 
    legend('Low','High','Location','northwest')
    title(cond_label{cond})
end
exportgraphics(gcf,[output_dir '\MOD_amp_GA_Fz.jpg']);
exportgraphics(gcf,[output_dir '\MOD_amp_GA_Fz.pdf']);
close;
