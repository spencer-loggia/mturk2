function convertStimSpaceForTablets(sname_f, fname_f, lumSD)
	
	
	tosave = [];  % suppresses erroneous errors
	
	if ~nargin
		nms = getSpaceShapeNames();
		
		sname_f = nms{1};  % rewards according to shapes & colors
		fname_f = nms{2};  % frequency of stimuli appearing
		lumSD = 5;  % decided upon
	elseif nargin == 1
		fname_f = sname_f{2};
		sname_f = sname_f{1};
		lumSD = 5;  % decided upon
    end
	
	% lspace
	% SD is expressed in GRID units - in reward and/or frequency space
	%  We desire +/- 10 Luminances, so 21 in total.
    %  This means we need 20 numbers (21 intervals)
    %  or 19 numbers on either side of center
    numLum = 21;
    numLim = (numLum-2)/2;
	lumX = -numLim:numLim;
	
	lumy = normcdf(lumX, 0, lumSD);
	
	
	
	sname = load(sname_f, '-mat');
	nlvl = sname.tosave(1);
	vlvl = sname.tosave(1+(1:nlvl));
	vdat = sname.tosave((2+nlvl):end);
	
	levels = vlvl;
	
	fname = load(fname_f, '-mat');
	nlvl = fname.tosave(1);
	plvl = fname.tosave(1+(1:nlvl));
	pdat = fname.tosave((2+nlvl):end);
		
	%  get the stimulus value space
	
	normalizedShapeSpace = MturkHelper.sumToOne(shapeSpace2D(vdat));
	rawRewardSpace = MturkHelper.convertToLevels(MturkHelper.maxToOne(normalizedShapeSpace), levels);
	rspace = rawRewardSpace(:);
	
	normalizedFreqSpace = MturkHelper.sumToOne(shapeSpace2D(pdat));
	fspace = normalizedFreqSpace(:);
	
    save([fname_f,'_freq'], 'fspace');
    save([sname_f,'_rewd'], 'rspace');
	
% 	outfilename = ['mturk2spaces_',num2str(now),'.txt'];
    base_name = fileparts(sname_f);
    outfilename = ['mturk2spaces_', '.txt'];
	fid = fopen(outfilename, 'w');
	disp(outfilename);
	
	fprintf(fid, '//// space name = %s\r\n', sname_f);
	fprintf(fid, '//// freq name = %s\r\n', fname_f);
	fprintf(fid, '//// Lum SD = %d, %d values\r\n', [lumSD, length(lumX)]);
	
	%%% Frequency space
	fprintf(fid, '"frequencySpace":[');
	
	ns = length(fspace);
	for n = 1:(ns-1)
		val = fspace(n);
		fprintf(fid, '%1.9f, ', val);
		
	end;
	val = fspace(ns);
	fprintf(fid, '%1.9f];\r\n', val);
	
	%%% individual freq file 
    ft = array2table(fspace);
    ft.Properties.VariableNames(1) = {'prob',};
    writetable(ft, 'freq_space.csv')
    
    %%% individual reward space
    rt = array2table(rspace);
    rt.Properties.VariableNames(1) = {'reward',};
    writetable(rt, 'reward_space.csv')
    
	%%% rewards for shapes
	fprintf(fid, '"rewardSpace":[');
	
	ns = length(rspace);
	for n = 1:(ns-1)
		val = rspace(n);
		fprintf(fid, '%d, ', val);
		
	end;
	val = rspace(ns);
	fprintf(fid, '%d];\r\n', val);
	
	
	%%% SD for luminance contrasts
	fprintf(fid, '"lumContrast":[');
	
	ns = length(lumy);
	for n = 1:(ns-1)
		val = lumy(n);
		fprintf(fid, '%1.9f, ', val);
		
	end;
	val = lumy(ns);
	fprintf(fid, '%1.9f];\r\n', val);
	
	
	
	fclose(fid);
	done;
	