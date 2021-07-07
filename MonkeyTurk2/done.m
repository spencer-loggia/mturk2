function done()

global ptime__global__var__ pname__global__var__

[stk, ~] = dbstack();

if length(stk) < 2
	pnm = '';
	etstr = '';
else
	pnm = stk(2).name;
	
	if strcmp(pnm, pname__global__var__)
		et = toc(ptime__global__var__);
		
		if et > 60
			et = et/60;
			unitstr = 'min';
		else
			unitstr = 'sec';
		end;
		
		etstr = ['     elapsed time: ', num2str(et), ' ', unitstr];
	else
		etstr = '';
	end;
end;

disp([pnm, ' done    ', datestr(clock), etstr]);


