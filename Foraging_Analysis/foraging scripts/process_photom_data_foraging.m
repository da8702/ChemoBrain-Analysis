function aligned_photo = process_photom_data_foraging(fname, subj_str, outdir)
    load(fname);
    [Photo, Time] =demod_and_baseline(SessionData);
    aligned_photo = align_photom_foraging(SessionData, Photo, Time);
    
    %plot a raster of the session, useful for checking that the photometry
    %wasnt garbage
    figure
    imagesc(aligned_photo)
    title(fname)

    [status, msg, msgID] = mkdir(outdir);
    save(fullfile(outdir,  subj_str + "_AlignedPhoto.mat"),'aligned_photo')
end