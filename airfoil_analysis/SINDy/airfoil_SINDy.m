DIR = 'C:\Users\Sohaib Bhatti\Documents\GitHub\data_driven_control\airfoil_analysis\proper_orthogonal_decomposition';

paramsFile = fullfile(DIR,'temporal_amplitudes.h5');

temporal_amplitudes = h5read(paramsFile,'/temporal_amplitudes'); 



for 1:length(temporal_amplitudes)
    Theta = poolData(temporal_amplitudes(i),n,3);
    sparsifyDynamics()

function Xi = sparsifyDynamics(Theta, dXdt,lambda, n)
    % Compute Sparse regression: sequential least squares
    Xi = Theta\dXdt;
    % Initial guess: Least-squares
    % Lambda is our sparsification knob.
    for k=1:10
        smallinds = (abs(Xi)<lambda);
        % Find small coefficients
        Xi(smallinds)=0;
        % and threshold
        for ind = 1:n
            % n is state dimension
            biginds = ~smallinds(:,ind);
            % Regress dynamics onto remaining terms to find sparse Xi
            Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind);
        end
    end
end