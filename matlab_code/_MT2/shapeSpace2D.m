function z = shapeSpace2D(nx, ny, varargin)
	
	
	% each Gaussian comprises 5 parameters:
	%   mu1, mu2, sd1, sd2, covar
	% Gaussian means, sigmas, and covars are all proportional to the range.
	%  They take on values 0-1, and the covar value is insured to make
	%  a valid covar matrix
	
	if nargin == 1
		% unpack the array;
		args = nx;
		nx = args(1);
		ny = args(2);
		varargin = args(3:end);
	end;

	nparm = 5;
	
	x = 1:nx;
	y = 1:ny;
	
	% add a margin all around
	x = -nx:2*nx;
	y = -ny:2*ny;	
	
	[xx, yy] = meshgrid(x, y);
	
	z = zeros(size(xx));

	varargin = num2cell(varargin);
	
	if mod(length(varargin), nparm)
		error('wrong number of params');
	else
		numGauss = (length(varargin)-2)/nparm;
	end;
	
	
	
	for g = 1:numGauss                           
		offset = 1 + nparm*(g-1);
		
		mu1 = scaleMu(varargin{offset}, x);
		mu2 = scaleMu(varargin{offset+1}, y);
		sd1 = scaleSigma(varargin{offset+2}, x);
		sd2 = scaleSigma(varargin{offset+3}, y);
		
		cvx = sqrt(sd1*sd2) - 1e-2;
		cvr = scaleVal(varargin{offset+4}, [-cvx, cvx]);
		
		mu = [mu1, mu2];
		sigma = [sd1, cvr; cvr, sd2];
		
		% make the space
		try
			d = mvnpdf([xx(:), yy(:)], mu, sigma);
% 			xOverlap = mvnpdf([xx(:)+xx(1), yy(:)], mu, sigma);
% 			yOverlap = mvnpdf([xx(:), yy(:)+yy(1)], mu, sigma);
		catch
			xx
			yy
			mu
			sigma
			errorok
		end;
		
		
		d = reshape(d, size(xx));
		d = d./max(d(:));
		
		if (any(isnan(d)))
			errorok;
		end;
		
		z = z + d;
	end;  % numGauss
	
	% wrap and trim the borders
	idx = 2:(nx+1);
	leftx = idx;
	centerx = leftx+nx;
	rightx = centerx + nx;
	
	idx = 2:(ny+1);
	lefty = idx;
	centery = lefty+nx;
	righty = centery + nx;
	

	z(:, centerx) = z(:, centerx) + z(:, leftx);
	z(:, centerx) = z(:, centerx) + z(:, rightx);
	
	z(centery, :) = z(centery, :) + z(lefty, :);
	z(centery, :) = z(centery, :) + z(righty, :);
	
	z = z(:, centerx);
	z = z(centery, :);
	
	
function val = scaleVal(v, x)
	val = x(1) + v * range(x);
	
	
	% dividing by three happens here because we have duplicated each dimension twice for wrap-around
function val = scaleMu(v, x)
	val = v * range(x)/3;
	
function val = scaleSigma(v, x)
	val = v * range(x)/3;
	
function gaussFromG(mu1, mu2, sd1, sd2, covar)
	
	