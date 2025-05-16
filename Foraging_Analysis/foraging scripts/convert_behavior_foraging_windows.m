
raw_behav_dir = "E:/";
subject_names = ["PBSNR01-LH"];
sub_dir = "Randelay_photo_singlevalue/Session Data";

%need to comment out NidaqData and Nidaq2Data in makeTrialEventsForaging.m
%when using for Randelay_changeover (no photometry data)
% sub_dir = "Randdelay_changeover/Session Data"; 

outdir = "E:/processed";


for i = 1:size(subject_names, 2)
    subj_name = subject_names(i);
    full_dir = fullfile(raw_behav_dir, subj_name, sub_dir, "/*.mat");
    listing = dir(full_dir);
    
    for j = 1: size(listing, 1)
        fname  = listing(j).name;
        % date = extractDatetime(fname);
        date = listing(j).date;
        date = strrep(date, ":", "-"); % Replace invalid characters
        process_photom_data_foraging(fullfile(raw_behav_dir, subj_name, sub_dir,fname), date, fullfile(outdir, subj_name));
        makeTrialEventsForaging(fullfile(raw_behav_dir, subj_name, sub_dir,fname), date, fullfile(outdir, subj_name))
    end

end