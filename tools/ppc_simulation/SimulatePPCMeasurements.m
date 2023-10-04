function SimulatePPCMeasurements(startidx, endidx, dataset) 
	if strcmp(dataset, "sunrgbd")
		base_dirpath = '../../data/sunrgbd/sunrgbd_trainval/';
		scenedir = fullfile(base_dirpath, 'image');
		depthdir = fullfile(base_dirpath, 'depth');
		out_base_dirpath = fullfile(base_dirpath, 'processed_lowfluxlowsbr_min2');
		num_bins = 1024; 
	elseif strcmp(dataset, "kitti")
		base_dirpath = '../../data/kitti/training/';
		depthdir = fullfile(base_dirpath, 'velodyne_reduced/');
		out_base_dirpath = fullfile(base_dirpath, 'processed_velodyne_reduced_lowfluxlowsbr2048_r025_dist10');
		num_bins = 2048; 
	end
	%out_base_dirpath = fullfile(base_dirpath, 'processed_testing');


	% Speed of light
	c = 3e8; 
	repetition_period = 600e-9;
	bin_size = repetition_period/num_bins; %approximately the bin size (in secs)

	% Dark and Bright image parameter idx
	dark_img_param_idx = 0; % For dark count image
	psf_img_param_idx = 0; % For bright image from which the PSF/pulse wavefor is extracted (see LoadAndPreprocessBightPSFImg.m for options)

	% Not all scenes have the same dimentions, we modify nr and nc in innner loop.
	% Just creating here to create a dataset name similar to original training set.
	nr = 576; nc = 704;
	
	% Create output directory
	sim_param_str = ComposeSimParamString(nr, nc, num_bins, bin_size, dark_img_param_idx, psf_img_param_idx);

	
	if strcmp(dataset, "sunrgbd")
		startdataidx=1; enddataidx=10335;
		simulation_params = [ 1, 100; 1, 50; 5, 100; 5, 50;];
	elseif strcmp(dataset, "kitti")
		startdataidx=0; enddataidx=7480;
		simulation_params = [ 5, 100; 5 250; 5 500; 5 1000];
	end

	% Get all scene names. they go from 000001 to 010335
	scenes=arrayfun(@(a)num2str(a,'%06d'),[startdataidx:enddataidx],'uni',0);
	scenes=convertCharsToStrings(scenes);
	%scenes = readlines(fullfile(base_dirpath, 'all_data_idx.txt'),"EmptyLineRule","skip");

	% Sensor parameters from https://github.com/facebookresearch/omnivore/issues/12
	% don't need these values for now.
	% datasets = ["kv1", "kv1_b", "kv2", "realsense", "xtion"];
	% baselines = [0.075, 0.075, 0.075, 0.095, 0.095];
	% sensor_to_params = dictionary(datasets, baselines)

	startidx = max(startidx, 1);
	endidx = min(endidx, length(scenes));

	outdir = fullfile(out_base_dirpath, sprintf('%s', sim_param_str));
	if ~exist(outdir, 'dir')
	    mkdir(outdir)
	end

	t_s = tic;
	for ss = startidx:endidx
	    fprintf('Processing scene %s...\n',scenes(ss));
	    out_fname = sprintf('spad_%s_%s_%s.mat', scenes(ss), num2str(simulation_params(end,1)), num2str(simulation_params(end,2)));
	    out_fpath = fullfile(outdir, out_fname)
	    if exist(out_fpath)
		    continue
	    end
	
	    if strcmp(dataset, "sunrgbd")
		% camera intrinsics
	    	K = dlmread(fullfile(base_dirpath, 'calib', sprintf('%s.txt', scenes(ss) )));
	    	allK = K;
	    	fx = K(2, 1); fy = K(2, 5);
	    	cx = K(2, 7); cy = K(2, 8);

		% read depth file
	    	dmin = 0.;
	    	depth = (single(imread(fullfile(depthdir, sprintf('%s.png', scenes(ss))) ) ) + dmin);
	    	depth = depth./1000;
		% read intensity image
	    	intensity = rgb2gray(im2double(imread(fullfile(scenedir, sprintf('%s.jpg', scenes(ss))))));   

	    	max_scene_depth = max(depth(:));
	    	min_scene_depth = min(depth(:));
	    	fprintf('    Max scene depth: %f...\n',max_scene_depth);
	    	fprintf('    Min scene depth: %f...\n',min_scene_depth);

	    	%mask = depth == dmin;
	    	% inpainting very small depth values too along with nan
	    	% because for some pixels, small depths results in very high signal ppp as it is alpha/dist^2
	    	% so removing about first 22 bins here
	    	mask = depth <= (dmin+2.);
	    	depth(mask) = nan;
	
	    	depth = full(inpaint_nans(double(depth),5));
	
	    	depth = max(depth, 0);
	    	intensity = max(intensity, 0);

		% convert depth to distance
	    	[x,y] = meshgrid(1:size(intensity,2),1:size(intensity,1));
	    	X = (x - cx).*depth ./ fx;
	    	Y = (y - cy).*depth ./ fy;
	    	dist = sqrt(X.^2 + Y.^2 + depth.^2);
	    	clear x1 x2 y1 y2 X1 X2 Y1 Y2;
	
	    	% set a number of signal photons per pixel
	    	albedo = intensity;
	    	alpha = albedo .* 1./ dist.^2;

	    elseif strcmp(dataset, "kitti")
	    	fid = fopen(fullfile(depthdir, sprintf('%s.bin', scenes(ss))));
	    	xyzr = double(fread(fid, '*float'));
	    	fclose(fid);

	    	X = xyzr(1:4:end);
	    	Y = xyzr(2:4:end);
	    	Z = xyzr(3:4:end);
	    	r = xyzr(4:4:end);
	    	[az, el, dist] = cart2sph(X, Y, Z);

	    	max_scene_dist = max(dist(:));
	    	min_scene_dist = min(dist(:));
	    	fprintf('    Max scene dist: %f...\n',max_scene_dist);
	    	fprintf('    Min scene dist: %f...\n',min_scene_dist);

	    	% adding 0.25 to reflectance values and 10 to distances to avoid 0 reflectance and small distances
	    	alpha = r + 0.25;
	    	intensity = alpha .* (dist+10).^2;
	    end

	    nr = size(dist, 1); nc = size(dist, 2);

	    % Load the dark count image. If dark_img_param_idx == 0 this is just zeros
	    % the dark image tells use the dark count rate at each pixel in the SPAD
	    dark_img = LoadAndPreprocessDarkImg(dark_img_param_idx, nr, nc);
	    
	    % Load the per-pixel PSF (pulse waveform on each pixel).
	    % For the simulation, we will scale each pixel pulse (signal), shift it
	    % vertically (background level), and shift it horizontally (depth/ToF).
	    [PSF_img, psf, pulse_len] = LoadAndPreprocessBrightPSFImg(psf_img_param_idx, nr, nc, num_bins);
	    %psf_data_fname = sprintf('PSF_used_for_simulation_nr-%d_nc-%d.mat', nr, nc);
	    %psf_data_fpath = fullfile(outdir, psf_data_fname);
	    %if ~exist(psf_data_fpath)
	    %	save(psf_data_fpath, 'PSF_img', 'psf', 'pulse_len');
	    %end
	
	    for zz = 1:size(simulation_params,1) 
	               
	        % Select the mean_signal_photons and mean_background_photons
	        mean_signal_photons = simulation_params(zz,1);
	        mean_background_photons = simulation_params(zz,2);
	        SBR = mean_signal_photons ./ mean_background_photons;
	        disp(['The mean_signal_photons: ', num2str(mean_signal_photons), ', mean_background_photon: ', ...
	            num2str(mean_background_photons), ', SBR: ', num2str(SBR)]);
	
	        % Simulate the SPAD measuements at the correct resolution
	        [spad, detections, rates, range_bins] = SimulateSPADMeasurement(alpha, intensity, dist, PSF_img, bin_size, num_bins, nr, nc, mean_signal_photons, mean_background_photons, dark_img, c);
	        
	        % save sparse spad detections to file
	        % the 'spad' is the simulated data, 'depth' is the GT 2D depth map,
	        % 'intensity' is the gray image, 'rates' is actually the GT 3D histogram.
	        out_fname = sprintf('spad_%s_%s_%s.mat', scenes(ss), num2str(mean_signal_photons), num2str(mean_background_photons));
	        out_fpath = fullfile(outdir, out_fname);
	        if strcmp(dataset, "sunrgbd")
	            SaveSimulatedSPADImgSUNRGBD(out_fpath, spad, SBR, range_bins, intensity, bin_size, allK, num_bins)
	        elseif strcmp(dataset, "kitti")
	            SaveSimulatedSPADImgKITTI(out_fpath, spad, SBR, range_bins, intensity, bin_size, num_bins, az, el, r)
	        end
	    end
	end
	t_cost = toc(t_s);
	disp(['Time cost: ', num2str(t_cost)]);
end
