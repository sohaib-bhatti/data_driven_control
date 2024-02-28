function [Phi, Lambda,Atilde,Amplitudes] = calc_dmd(Data,r)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Perform Dynamic Mode Decomposition
%
% Inputs:
% Data: A matrix with snapshots (equispaced in time) as columns
% (can be mean-subtracted)
% r: Truncation level of SVD (in effect, Atilde will be the distrete-time
% system matrix in the space of the first r POD coefficients).
%
% Outputs:
% Phi: DMD modes
% Lambda: DMD eigenvalues
% Atilde: DMD matrix (i.e., matrix governing the evolution of discrete-time
%         dynamical system, in space of POD coefficients)
% Amplitudes: Amplitudes of DMD modes (note one of many possible
%             definitions is used to define these)
% 
% Note that this method assumes that the inner product is defined by 
% <x,y> = x^*y. To apply a different inner product, the code would need to
% be modified, or the input data appropriately scaled.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X1 = Data(:,1:end-1);
X2 = Data(:,2:end);

[U,S,V] = svd(X1,'econ');
if r ~= 0
    S = S(1:r,1:r);
    U = U(:,1:r);
    V = V(:,1:r);
end

D = diag(S);
Dinv = 1./D;
Sinv = diag(Dinv);

% Atilde is the identified discrete-time operator governing the dynamics,
% in POD coordinates
Atilde = U'*X2*V*Sinv;
[Vtilde, Lambda] = eig(Atilde);
Phi = X2*V*Sinv*Vtilde;

Amplitudes = (Phi*Lambda)\Data(:,1); %note that there are several ways to define mode amplitudes

% sort based on Amplitude
[~,SortInds] = sort(abs(Amplitudes),'descend');
Amplitudes= Amplitudes(SortInds);
Lambda = diag(Lambda);
Lambda = Lambda(SortInds);
Phi = Phi(:,SortInds);
