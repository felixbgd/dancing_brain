%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Imaging the dancing brain" paper - Bigand et al. (2024) %
%             Kinematic feature (PM) selection             %
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
addpath(['.\mtrf\']); 
ft_defaults;


%% Set parameters, folders, paths etc

output_dir ='.\results_step1_PMselection\';
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
% Audio waveforms --> for full model
stim_style = 'all';
predictor_audio = 'StimAcoustics';
allpred_audio = load(['.\\' predictor_audio]);

% Principal Movements (PMs)) --> for self, other models
predictor_motion_pm = 'StimMotion_pms';
allpred_motion_pm = load(['.\\' predictor_motion_pm]);

% Audio Lag  (s)
audioLag = 0.0535;

% TRF model parameters
minlag = -250;
maxlag = 300;
lambdas = [0 10.^(-4:8)]; 

% Loop for training subject-specific TRF models
Stim_trialsCond_audio=[]; Stim_trialsCond_motion=[];  EEG_trialsCond=[];
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
    data_bp_rs_asr_ica_interp           = Giac_removeChannels(data_bp_rs_asr_ica_interp_all,{'EXG1','EXG2','EXG3','EXG4','EXG5','EXG6'});
    [labels_sorted , idx_labels_sorted] = sort(data_bp_rs_asr_ica_interp.label);

    %%% Define pred/stim and EEG for each trial
    % Find trials corresponding to each condition
    for cond=1:4  tr_cond{cond} = find(tr==cond)';      end
    
    % Exception message for one trial (the last one) of one dyad absent
    if ismember(p,[73,74])
        tr_missing=32;
        tr_nb = NB_TRIALS - 1;
    else
        tr_nb = NB_TRIALS;
    end
    
    % Loop over
    iTrial = 1;
    for ii = 1:tr_nb       % For each trial that you wanna analyse
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
        
        % Define trials of, and trim, EEG data
        cfg              = [];
        cfg.trials       = ii; 
        EEG_of_one_trial = ft_selectdata(cfg,data_bp_rs_asr_ica_interp);
            
        startSong = startSong + prestim + EEG_of_one_trial.sampleinfo(1);     % Add the timeframe of sampleinfo
    
        cfg = [];
        cfg.trl(:,1) = startSong + cutOnset;        % start 
        cfg.trl(:,2) = startSong + lensong;        % stop 
        cfg.trl(:,3) = 0;        
        
        % Trim/cut EEG
        ft_EEG_of_one_trial_cut = ft_redefinetrial(cfg,EEG_of_one_trial);
        EEG_of_one_trial_cut = ft_EEG_of_one_trial_cut.trial{1,1};

        % Reorder the channels labels for EEG data (this is necessary
        % because of interpolation step in prepro)
        EEG_of_one_trial_cut = EEG_of_one_trial_cut(idx_labels_sorted,:); 
        eeg_label_reorder = {ft_EEG_of_one_trial_cut.label{idx_labels_sorted}};

        %%% Store EEG per trial
        EEG_trialsCond{iTrial} = EEG_of_one_trial_cut;
        
        %%% Store predictors/stimuli per trial %%%
        %%% Audio/music %%%
        pred_idx = find(strcmp(allpred_audio.stim_names,['song' num2str(songNum_self) '_' stim_style '.wav']));
        Stim_one_trial_audio = [allpred_audio.specflux_avg{pred_idx}']; sz_feat=[1]; feat_names={'T specflux'};
        Stim_one_trial_audio = Stim_one_trial_audio(:,cutOnset:lensong);

        %%% Self %%%
        iSelfParticipant = p; NB_PM = 15;
        Stim_one_trial_motion_self_signed = cell2mat(allpred_motion_pm.motion_pms{iSelfParticipant,ii});
        Stim_one_trial_motion_self_signed = Stim_one_trial_motion_self_signed(:,cutOnset:lensong);
        Stim_one_trial_motion_self_signed = Stim_one_trial_motion_self_signed(1:NB_PM,:);
        Stim_one_trial_motion_self        = abs(Stim_one_trial_motion_self_signed);

        %%% Other %%%
        Stim_one_trial_motion_other_signed = cell2mat(allpred_motion_pm.motion_pms{iOtherParticipant,ii}); 
        Stim_one_trial_motion_other_signed = Stim_one_trial_motion_other_signed(:,cutOnset:lensong);
        Stim_one_trial_motion_other_signed = Stim_one_trial_motion_other_signed(1:NB_PM,:);
        Stim_one_trial_motion_other        = abs(Stim_one_trial_motion_other_signed);

        % Padding motion signals with 0 (to avoid edge effects)
        Stim_one_trial_motion_self(:,1:100) = 0;
        Stim_one_trial_motion_self(:,end-100:end) = 0;
        Stim_one_trial_motion_other(:,1:100) = 0;
        Stim_one_trial_motion_other(:,end-100:end) = 0;
        
        % Store per trial
        Stim_trialsCond_audio{iTrial} = Stim_one_trial_audio;
        Stim_trialsCond_motion_self{iTrial} = Stim_one_trial_motion_self;
        Stim_trialsCond_motion_other{iTrial} = Stim_one_trial_motion_other;
        
        iTrial = iTrial+1;
    end
    
    % Exception message for one trial (the last one) of one dyad absent
    if ismember(p,[73,74])
        Stim_trialsCond_audio{tr_missing} = [];
        Stim_trialsCond_motion_self{tr_missing} = [];
        Stim_trialsCond_motion_other{tr_missing} = [];
        EEG_trialsCond{tr_missing} = [];
    end


    %%% Normalisation of EEG and stimuli/predictors (see Crosse et al., 2021) %%%%
    %%% Normalize them based on "global std" of the participant               %%%%
    % Compute "global std" of predictors/stimuli
    Stim_allTrials_concat_audio = [Stim_trialsCond_audio{:}];
    Stim_audio_global_std = sqrt(mean(var(Stim_allTrials_concat_audio,[],2)));

    Stim_allTrials_concat_motion_self = [Stim_trialsCond_motion_self{:}];
    Stim_motion_self_local_std = std(Stim_allTrials_concat_motion_self,[],2);

    Stim_allTrials_concat_motion_other = [Stim_trialsCond_motion_other{:}];
    Stim_motion_other_local_std = std(Stim_allTrials_concat_motion_other,[],2);

    % Compute "global std" of EEG
    EEG_allTrials_concat = [EEG_trialsCond{:}];
    EEG_global_std = sqrt(mean(var(EEG_allTrials_concat,[],2)));

    % Normalize, and create stim/pred cell arrays for full and reduced models
    for iTrial=1:tr_nb
        Stim_trialsCond_audio{iTrial} = (Stim_trialsCond_audio{iTrial} ./ Stim_audio_global_std)';
        Stim_trialsCond_motion_self{iTrial} = (Stim_trialsCond_motion_self{iTrial} ./ Stim_motion_self_local_std)';   % local or global?
        Stim_trialsCond_motion_other{iTrial} = (Stim_trialsCond_motion_other{iTrial} ./ Stim_motion_other_local_std)';   % local or global?
        
        Stim_trialsCond_full{iTrial} = [Stim_trialsCond_audio{iTrial} , Stim_trialsCond_motion_self{iTrial},Stim_trialsCond_motion_other{iTrial}];
        for pm = 1:NB_PM
            pms_remaining = setxor(pm,1:15);
            Stim_trialsCond_minus_self{iTrial,pm} = [Stim_trialsCond_audio{iTrial} , Stim_trialsCond_motion_self{iTrial}(:,pms_remaining) , Stim_trialsCond_motion_other{iTrial}];
            Stim_trialsCond_minus_other{iTrial,pm} = [Stim_trialsCond_audio{iTrial} , Stim_trialsCond_motion_self{iTrial} , Stim_trialsCond_motion_other{iTrial}(:,pms_remaining)];
        end
         EEG_trialsCond{iTrial} = (EEG_trialsCond{iTrial} ./ EEG_global_std)';
    end
    
    % Store the pred/stim and EEG per participant and condition --> for the
    % generic approach when we test generic TRF on each participant (later)
    for cond=1:4
        trials_of_cond = tr_cond{cond};
        % Exception message for one trial (the last one) of one dyad absent
        if ismember(p,[73,74]) && length(find(trials_of_cond==32))>0    
            cond_missing = cond;  tr_missing_in_cond=find(tr_cond{cond}==32);
            trials_of_cond(tr_missing_in_cond)=[]; 
        end

        Stim_part_full{cond,iParticipant} = {Stim_trialsCond_full{trials_of_cond}};
        for pm = 1:NB_PM
            Stim_part_minus_self{cond,iParticipant,pm} = {Stim_trialsCond_minus_self{trials_of_cond,pm}};
            Stim_part_minus_other{cond,iParticipant,pm} = {Stim_trialsCond_minus_other{trials_of_cond,pm}};
        end
        EEG_part{cond,iParticipant} = {EEG_trialsCond{trials_of_cond}};
    end
   
    %%%%%%%%%%%%%%%%%
    %%%% RUN TRF %%%%
    %%%%%%%%%%%%%%%%%
    nb_models = 31;  % every PM self/other + full
    nb_feat = [31 , 30.*ones(1,30)];
    for feat=1:nb_models
        if feat==1 Stim = Stim_trialsCond_full; conditions_to_fit=[1:4]; end
        if ismember(feat,[2:16]) Stim = {Stim_trialsCond_minus_self{:,feat-1}}; conditions_to_fit=[3:4]; end
        if ismember(feat,[17:31]) Stim = {Stim_trialsCond_minus_other{:,feat-16}}; conditions_to_fit=[1:2]; end
        
        %%% Train one TRF per condition and for full+reduced models
        cv_avgtr=[]; cv_avgtrch=[]; opt_idx=[]; opt_lmb=[]; mdl_test=[]; st_test=[];
        for cond=conditions_to_fit
            trials_of_cond = tr_cond{cond};
            
            % Exception message for one trial (the last one) of one dyad absent
            if ismember(p,[73,74]) && length(find(trials_of_cond==32))>0    
                cond_missing = cond;  tr_missing_in_cond=find(tr_cond{cond}==32);
                trials_of_cond(tr_missing_in_cond)=[]; 
            end
        
            for test_tr=trials_of_cond
                disp(['TRF participant ' num2str(iParticipant) ' - feat ' num2str(feat)])
                % Leave-one-trial-out cross-validation (CV) training/testing
                train_tr = setxor(trials_of_cond,test_tr);
                Stim_cond = {Stim{:}}';
                EEG_cond = {EEG_trialsCond{:}}';
                stats = mTRFcrossval(Stim_cond(train_tr),EEG_cond(train_tr),fps,1,minlag,maxlag,lambdas,'verbose',0);
                
                % Find optimal lambda out of the CV folds
                % average over CV folds
                cv_avgtr = squeeze(mean(stats.r,1));
                % average over channels
                cv_avgtrch(:,test_tr) = mean(cv_avgtr,2)';
                % identify the optimal lambda
                opt_idx = find(cv_avgtrch(:,test_tr)==max(cv_avgtrch(:,test_tr)),1,'first'); % if multiple lambdas have identical performance, pick the smallest lambda
                opt_lmb(test_tr) = lambdas(opt_idx);

                % Fit the model on all training trials
                mdl_test = mTRFtrain(Stim_cond(train_tr),EEG_cond(train_tr),fps,1,minlag,maxlag,opt_lmb(test_tr));
                % Test on the left-out testing trial
                [~,st_test] = mTRFpredict(Stim_cond(test_tr),EEG_cond(test_tr),mdl_test);
                
                % Store TRF model per trial
                r_test(feat,test_tr,:)                   = st_test.r;  % TRF prediction (r) 
                w_test(feat,test_tr,1:nb_feat(feat),:,:) = mdl_test.w; % TRF model weights  
                b_test(feat,test_tr,:)                   = mdl_test.b; % TRF model intercept
            end
        end
    end

    %%% Store TRF model per condition (instead of trial)
    r_cond = [];  w_cond = [];  b_cond=[]; 
    for feat=1:nb_models
        if feat==1 conditions_to_fit=[1:4]; end
        if ismember(feat,[2:16]) conditions_to_fit=[3:4]; end
        if ismember(feat,[17:31]) conditions_to_fit=[1:2]; end
        for cond=conditions_to_fit
            trials_of_cond = tr_cond{cond};
            
            % Exception message for one trial (the last one) of one dyad absent
            if ismember(p,[73,74]) && length(find(trials_of_cond==32))>0    
                cond_missing = cond;  tr_missing_in_cond=find(tr_cond{cond}==32);
                trials_of_cond(tr_missing_in_cond)=[]; 
            end
            
            % Store
            for tr=1:length(trials_of_cond)
                % Genuine TRF
                r_cond(feat,cond,tr,:) = r_test(feat,trials_of_cond(tr),:);
                w_cond(feat,cond,tr,:,:,:) = w_test(feat,trials_of_cond(tr),:,:,:);
                b_cond(feat,cond,tr,:) = b_test(feat,trials_of_cond(tr),:);
            end
        end
    end

    % Average per cond
    r_cond_avg = squeeze(mean(r_cond,3));
    r_part(iParticipant,:,:,:) = r_cond_avg;
    w_cond_avg = squeeze(mean(w_cond,3));
    w_part(iParticipant,:,:,:,:,:) = w_cond_avg;
    b_cond_avg = squeeze(mean(b_cond,3));
    b_part(iParticipant,:,:,:) = b_cond_avg;
    
    % Interim plot of the subject-specific TRF weights
    chans_to_plot={'Fz','Cz','Oz'};
    for chan=chans_to_plot
        chan=cell2mat(chan);
        ch=find(strcmp(eeg_label_reorder,chan));
        idxChan = ch;
        cond_label = {'YV-SM','YV-DM','NV-SM','NV-DM'};
        figure;
        set(gcf, 'Position',  [0, 0, 2800, 1000])
        
        absMaxZ = max(abs(w_cond_avg(1,:,:,:,idxChan)),[],'all');        
        for cond=1:4
            nexttile
            
            for feat=1:nb_feat(1)
                plot(squeeze(w_cond_avg(1,cond,feat,:,idxChan)),'LineWidth',1.3); hold on
            end
            ylim([-absMaxZ*1.3 absMaxZ*1.3])
            xlim([1 length(mdl_test.t)])
            xticks(1:5:length(mdl_test.t)); xticklabels(mdl_test.t(1:5:end));
            if cond==4 legend(); end
            title(cond_label{cond},'FontSize',14);
        end
        exportgraphics(gcf,[output_dir '\INDIV_p' numParticipant '_weights_' chan '.jpg']);
        close;
    end


    iParticipant = iParticipant +1;
end

% Save weights and intercepts to store the subject-specific TRF models
save([output_dir 'TRFmodels'],'mdl_test','w_part','b_part');

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         PLOT GRAND AVERAGE TRF WEIGHTS         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% UNCOMMENT IF YOU WANT TO load weights and intercepts of previously stored TRF models
% w_part = zeros(70,31,4,31,61,64);
% b_part = zeros(70,31,4,64);
% for feat=1:31
% %     w_feat = squeeze(w_part(:,feat,:,:,:,:));
% %     b_feat = squeeze(b_part(:,feat,:,:));
% %     save([output_dir 'TRFmodels_' num2str(feat)],'mdl_test','w_feat','b_feat');
% 
%     load([output_dir 'TRFmodels_' num2str(feat) '.mat']);
%     w_part(:,feat,:,:,:,:) = w_feat;
%     b_part(:,feat,:,:) = b_feat;
%     clear w_feat
%     clear b_feat
% end


w_grandavg = squeeze(mean(w_part,1)); 
w_grandavg_stderr = squeeze(std(w_part,1)) ./ sqrt(size(w_part,1)); 

% Plot
chans_to_plot={'FCz','Cz','C5','C6','Oz'};
feats_to_plot_label={'Audio','Motion-self','Motion-other','Synchrony'};
colors = [ [0 0.447 0.741] ;[0.929 0.694 0.125];[0.466 0.674 0.188];[0.2 0.54 0.6]];

for chan=1:length(chans_to_plot)
    ch=find(strcmp(eeg_label_reorder,chans_to_plot{chan}));
    idxChan = ch;
    cond_label = {'YV-SM','YV-DM','NV-SM','NV-DM'};
    figure;
    set(gcf, 'Position',  [0, 0, 1800, 1000])
    tiledlayout(2,4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for i=1:nb_feat(1)
        pred=i;
        absMaxZ = max(abs(w_grandavg(1,:,pred,10:end-10,idxChan)),[],'all');
        nexttile

        xval = 1:length(mdl_test.t);
        plot(squeeze(w_grandavg(1,1,pred,:,idxChan)),'LineWidth',1.3,'Color',colors(1,:));  fefe_plotShadedError(xval, squeeze(w_grandavg(1,1,pred,:,idxChan))', squeeze(w_grandavg_stderr(1,1,pred,:,idxChan))'); 
        plot(squeeze(w_grandavg(1,2,pred,:,idxChan)),'LineWidth',1.3,'Color',colors(2,:));  fefe_plotShadedError(xval, squeeze(w_grandavg(1,2,pred,:,idxChan))', squeeze(w_grandavg_stderr(1,2,pred,:,idxChan))'); 
        plot(squeeze(w_grandavg(1,3,pred,:,idxChan)),'LineWidth',1.3,'Color',colors(3,:));  fefe_plotShadedError(xval, squeeze(w_grandavg(1,3,pred,:,idxChan))', squeeze(w_grandavg_stderr(1,3,pred,:,idxChan))');
        plot(squeeze(w_grandavg(1,4,pred,:,idxChan)),'LineWidth',1.3,'Color',colors(4,:));  fefe_plotShadedError(xval, squeeze(w_grandavg(1,4,pred,:,idxChan))', squeeze(w_grandavg_stderr(1,4,pred,:,idxChan))');
        
        ylim([-absMaxZ*1.5 absMaxZ*1.5])
        xlim([1 length(mdl_test.t)])
        xticks(1:5:length(mdl_test.t)); xticklabels(mdl_test.t(1:5:end));
        if cond==4 legend(cond_label,'FontSize',12); end
        title(feats_to_plot_label{i},'FontSize',14); 
    end
    exportgraphics(gcf,[output_dir 'GDAVG_weights_' cell2mat(eeg_label_reorder(ch)) '.jpg']);
    exportgraphics(gcf,[output_dir 'GDAVG_weights_' cell2mat(eeg_label_reorder(ch)) '.pdf']);

    close;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     TEST TRF ON AVERAGE WEIGHTS OF N-1 SUBJ    %%%
%%%            TESTED ON 1 LEFTOUT SUBJ            %%%
%%%               (GENERIC APPROACH)               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TRF (only test, not train)
idxStart = 1;
idxStop = length(mdl_test.t);
for feat=1:nb_models
    if feat==1 
        load([output_dir 'Stim_part_full']);
        Stim = Stim_part_full; conditions_to_fit=[1:4]; 
    elseif ismember(feat,[2:16])
        pm=feat-1; load([output_dir 'Stim_part_minus_self_pm_' num2str(pm)]);
        Stim = Stim_part_minus_self_pm; conditions_to_fit=[3:4]; 
    elseif ismember(feat,[17:31]) 
        pm=feat-16; load([output_dir 'Stim_part_minus_other_pm_' num2str(pm)]);
        Stim = Stim_part_minus_other_pm; conditions_to_fit=[1:2]; 
    end
    for cond=conditions_to_fit
        r_opt=[]; r_opt_avg=[];
        disp(['TRF RUNNING - FEAT ' num2str(feat) ' - COND ' num2str(cond)])
        for p_test=1:iParticipant-1
            disp(['PARTICIPANT ' num2str(p_test)])

            % Create a TRF model based on average of N-1 subjects' TRF
            p_train = setxor(1:iParticipant-1,p_test);
            w_avgTrain = mean(w_part(p_train,feat,cond,1:nb_feat(feat),idxStart:idxStop,:),1);
            b_avgTrain = mean(b_part(p_train,feat,cond,:),1);

            mdl_subjTrain = mdl_test;     % Just create the structure of TRF model, then update it
            mdl_subjTrain.t = mdl_test.t(idxStart:idxStop);
            mdl_subjTrain.w = reshape(w_avgTrain,size(w_avgTrain,4:6)); 
            mdl_subjTrain.b = reshape(b_avgTrain,size(b_avgTrain,3:4));
   
            % Select Stim and EEG data of the 1 left-out subject
            Stim_cond = Stim{cond,p_test};
            EEG_cond = EEG_part{cond,p_test};
            
            % Predict EEG of this left-out subject from the averaged TRF of
            % N-1 subjects
            [~,st_test] = mTRFpredict(Stim_cond,EEG_cond,mdl_subjTrain);
            gen_r_part(feat,cond,p_test,:) = squeeze(mean(st_test.r));
        end
    end
end



%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  PLOT TOPOGRAPHIES OF DELTA R FOR EACH MODEL   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute delta r per participant
gen_diff_r_part_topo = squeeze(gen_r_part(1,:,:,:) - gen_r_part(2:end,:,:,:));

% Compute grand average
gen_diff_r_YV = squeeze(mean(gen_diff_r_part_topo(:,1:2,:,:),2));
gen_diff_r_NV = squeeze(mean(gen_diff_r_part_topo(:,3:4,:,:),2));
GA_gen_diff_r_part_YV = squeeze(mean(gen_diff_r_YV,2));
GA_gen_diff_r_part_NV = squeeze(mean(gen_diff_r_NV,2));

% Plot the SELF contribution (every topo is a PM, in the YV conditions)
figure;
set(gcf, 'Position',  [0, 0, 1800, 1000])
iPlot=1;
absMaxZ = max(abs(GA_gen_diff_r_part_NV(1:NB_PM,:)),[],'all');
for d=1:NB_PM
    subplot(2,round(NB_PM/2),iPlot);
    ft_corrNeck_grandavg = struct;
    ft_corrNeck_grandavg.freq = 1;
    ft_corrNeck_grandavg.label = eeg_label_reorder;
    ft_corrNeck_grandavg.powspctrm = squeeze(GA_gen_diff_r_part_NV(d,:));

    cfg = [];
    cfg.marker       = 'on';
    cfg.colorbar     = 'no';
    cfg.colormap     = '*RdBu';
    cfg.layout       = 'biosemi64';
    cfg.figure = 'gca';
    cfg.comment='no';
    cfg.zlim = [-absMaxZ absMaxZ];
    ft_topoplotTFR(cfg, ft_corrNeck_grandavg);
    title(['PM' num2str(d)],'FontSize',14); 
    iPlot=iPlot+1;
end
exportgraphics(gcf,[output_dir 'GENERIC_r_GDAVG_SELF.jpg']);
exportgraphics(gcf,[output_dir 'GENERIC_r_GDAVG_SELF.pdf']);
close;

% Plot the OTHER contribution (every topo is a PM, in the YV conditions)
figure;
set(gcf, 'Position',  [0, 0, 1800, 1000])
iPlot=1;
absMaxZ = max(abs(GA_gen_diff_r_part_YV(NB_PM+1:2*NB_PM,:)),[],'all');
for d=NB_PM+1:2*NB_PM
    subplot(2,round(NB_PM/2),iPlot);
    ft_corrNeck_grandavg = struct;
    ft_corrNeck_grandavg.freq = 1;
    ft_corrNeck_grandavg.label = eeg_label_reorder;
    ft_corrNeck_grandavg.powspctrm = squeeze(GA_gen_diff_r_part_YV(d,:));

    cfg = [];
    cfg.marker       = 'on';
    cfg.colorbar     = 'no';
    cfg.colormap     = '*RdBu';
    cfg.layout       = 'biosemi64';
    cfg.figure = 'gca';
    cfg.comment='no';
    cfg.zlim = [-absMaxZ absMaxZ];
    ft_topoplotTFR(cfg, ft_corrNeck_grandavg);
    title(['PM' num2str(d-15)],'FontSize',14); 
    iPlot=iPlot+1;
end
exportgraphics(gcf,[output_dir 'GENERIC_r_GDAVG_OTHER.jpg']);
exportgraphics(gcf,[output_dir 'GENERIC_r_GDAVG_OTHER.pdf']);
close;



%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  COMPARE CONTRIBUTION OF EACH PM, SELF/OTHER   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparisons of delta r across PMs
% delta r is computed at Cz for self models, and Oz for other models

% Compute delta r per participant
gen_diff_r_part_topo = squeeze(gen_r_part(1,:,:,:) - gen_r_part(2:end,:,:,:));

% Compute delta r at Cz and Oz respectively
ROI_self = 19;
ROI_other = 44;

for part=1:iParticipant-1
    % Compute average for YV and NV conditions
    gen_diff_r_YV = squeeze(mean(gen_diff_r_part_topo(:,1:2,part,:),2));
    gen_diff_r_NV = squeeze(mean(gen_diff_r_part_topo(:,3:4,part,:),2));
        
    % Compute the delta r on the electrodes of interest, for each PM
    for pm = 1:NB_PM
        gen_diff_r_part_av_SELF(pm,part)=squeeze(mean(gen_diff_r_NV(pm,ROI_self),2));
        gen_diff_r_part_av_OTHER(pm,part)=squeeze(mean(gen_diff_r_YV(pm+15,ROI_other),2));
    end
end

% PLOTS
% Compute grand average (GA) and standard error (SE) of delta r 
% (now 1bar = 1 feat/1 condition)
GA_gen_diff_r_av_SELF = squeeze(mean(gen_diff_r_part_av_SELF,2));
GA_gen_diff_r_av_OTHER = squeeze(mean(gen_diff_r_part_av_OTHER,2));

figure;
set(gcf, 'Position',  [0, 0, 1800, 1000])
bar(GA_gen_diff_r_av_SELF);
exportgraphics(gcf,[output_dir 'ROI_UNIQUE_contrib_SELF.jpg']);
close;

figure;
set(gcf, 'Position',  [0, 0, 1800, 1000])
bar(GA_gen_diff_r_av_OTHER);
exportgraphics(gcf,[output_dir 'ROI_UNIQUE_contrib_OTHER.jpg']);
close;


