function nms = getSpaceShapeNames(idx)
	
	% returns the parameter file names defining spaces for MonkeyTurk2
	%
	%  nms{1} is the continuous reward space (not converted to levels yet)
	%  nms{2} is the continuous frequency space (how often stims appear)
	
	sname = 'MTurkSpaceParams22075.609';  % rewards according to shapes & colors
	fname = 'MTurkSpaceParams22077.061';  % frequency of stimuli appearing
	
	defaultNames = {sname, fname};
	
	if ~nargin
		nms = defaultNames;
		return;
	end
	
	
	switch idx
		case 0,
			nms = defaultNames;
			
		case 1,
			nms = {'MTurkSpaceParams22098.324', 'MTurkSpaceParams22098.954'};
			
		case 2,
			nms = defaultNames;
			
		case 3,
			nms = defaultNames;
			
		case 4,
			nms = defaultNames;
			
		case 5,
			nms = defaultNames;
			
		case 6,
			nms = defaultNames;
			
		otherwise
			nms = defaultNames;
	end;
			