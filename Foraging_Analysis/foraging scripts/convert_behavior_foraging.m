raw_behav_dir = "/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/Foraging_Cisplatin";
subject_names = ["DA123", "DA124", "DA125", "DA126", ...
                "DA127", "DA128", "DA129", "DA130", ...
                "DA131", "DA132", "DA133", "DA134"];
sub_dir = "Randelay_photo_singlevalue/Session Data";

outdir = "/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/Foraging_Cisplatin/Cis4_processed";

% Add progress reporting
fprintf('Starting processing for %d subjects\n', length(subject_names));

for i = 1:length(subject_names)
    subj_name = subject_names(i);
    fprintf('\nProcessing subject: %s\n', subj_name);
    
    % Construct the full path to the Session Data directory
    full_dir = fullfile(raw_behav_dir, "Cis4", subj_name, sub_dir);
    fprintf('Looking for data in: %s\n', full_dir);
    
    % Check if directory exists
    if ~exist(full_dir, 'dir')
        fprintf('Warning: Directory does not exist: %s\n', full_dir);
        continue;
    end
    
    % Get list of .mat files
    listing = dir(fullfile(full_dir, "*.mat"));
    fprintf('Found %d .mat files\n', length(listing));
    
    % Filter out hidden files
    valid_files = listing(~startsWith({listing.name}, '._'));
    fprintf('After filtering hidden files: %d valid files\n', length(valid_files));
    
    if isempty(valid_files)
        fprintf('No valid files found for subject %s\n', subj_name);
        continue;
    end
    
    for j = 1:length(valid_files)
        fname = valid_files(j).name;
        fprintf('\nProcessing file %d/%d: %s\n', j, length(valid_files), fname);
        
        try
            % Process photometry data
            fprintf('Processing photometry data...\n');
            process_photom_data_foraging(fullfile(full_dir, fname), valid_files(j).date, fullfile(outdir, subj_name));
            fprintf('Photometry data processing completed\n');
            
            % Process behavior data
            fprintf('Processing behavior data...\n');
            makeTrialEventsForaging(fullfile(full_dir, fname), valid_files(j).date, fullfile(outdir, subj_name));
            fprintf('Behavior data processing completed\n');
            
            fprintf('Successfully processed file: %s\n', fname);
        catch e
            fprintf('Error processing file %s:\n', fname);
            fprintf('Error message: %s\n', e.message);
            fprintf('Error identifier: %s\n', e.identifier);
            if ~isempty(e.stack)
                fprintf('Error occurred in: %s, line %d\n', e.stack(1).name, e.stack(1).line);
            end
            fprintf('Continuing with next file...\n');
        end
    end
end

fprintf('\nProcessing complete!\n');