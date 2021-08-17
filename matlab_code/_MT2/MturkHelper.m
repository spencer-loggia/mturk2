classdef  MturkHelper < handle
	
	%  Methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	
	% 		function m = xMarginal(a)
	% 		function m = yMarginal(a)
	% 		function m = zMarginal(a)
	% 		function A = marginalPrediction(xm, ym, zm)
	% 		function z = sumToOne(a)
	% 		function z = maxToOne(a)
	% 		function z = convertToLevels(a, l)
	% 		function z = spaceDiffError(s1, s2)
	% 		function y = expEqn(x, z)
	% 		function y = gaussEqn(x, z)
	
	
	methods(Static)
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  constructor
		function RP = MturkHelper()
		end;  % constructor
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  xMarginal
		function m = xMarginal(a)
			m = squeeze(sum(a,3));
			m = MturkHelper.sumToOne(sum(m,1));
			% 	m = m(:)./numel(a(:,1,:));
			m = m(:);
		end;  % xMarginal
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  yMarginal
		function m = yMarginal(a)
			m = squeeze(sum(a,3));
			m = MturkHelper.sumToOne(sum(m,2));
			m = m(:);
		end;  % yMarginal
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  zMarginal
		function m = zMarginal(a)
			m = squeeze(sum(a,2));
			m = MturkHelper.sumToOne(sum(m,1));
			% 	m = m(:)./numel(a(:,:,1));
			m = m(:);
		end;  % zMarginal
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  marginalPrediction
		function A = marginalPrediction(xm, ym, zm)
			
			nx = length(xm);
			ny = length(ym);
			nz = length(zm);
			
			A = ones(ny, nx, nz);
			xyA = ym*xm';
			
			for z = 1:nz
				A(:,:,z) = xyA.*zm(z);
			end;
			
			A = MturkHelper.sumToOne(A);
		end;  % marginalPrediction
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  sumToOne
		function z = sumToOne(a)
			z = a./sum(a(:));
		end	% sumToOne
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  maxToOne
		function z = maxToOne(a)
			z = a./max(a(:));
		end	% maxToOne
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  convertToLevels
		function z = convertToLevels(a, l)
			z = zeros(size(a));
			
			l = sort(l);
			
			for x = 1:length(l)
				z(a>l(x)) = x;
			end;
			
		end	% convertToLevels
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  spaceDiffError
		function z = spaceDiffError(s1, s2)
			
			errspc = abs(MturkHelper.maxToOne(s1) - MturkHelper.maxToOne(s2));
			z = sum(errspc(:))/numel(s1);
			
			
		end	% spaceDiffError
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%%  functions for fitting
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		function y = expEqn(x, z)
			mu = z(1);
			offset = z(2);
			scale = z(3);
			
			y = exppdf(x, mu);
			y = y./max(y);
			y = offset + scale.*y;
			
		end; % expEqn
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		function y = gaussEqn(x, z)
			mu = z(1);
			sigma = z(2);
			offset = z(3);
			scale = z(4);
			
			y = normpdf(x, mu, sigma);
			y = y./max(y);
			y = offset + scale.*y;
			
		end; % expEqn
		
		
	end  % methods
	
	
end  % classdef
