import cvxopt
from cvxopt import matrix, spmatrix, spdiag, sqrt, div

def interior_point( \
    ## System matrices
    H, g, \
    ## Equality constraint matrices
    A, b, \
    ## Inequality constraint matrices
    C, d, \
    ## Initial guess
    x_0, y_0, z_0, s_0,
    ## ALgirithm parameters
    eta=0.995, tol=1e-10, maxiter=10, ):


    ## Type checking
    H = matrix( H )
    g = matrix( g )
    A = matrix( A )
    b = matrix( b )
    C = matrix( C )
    d = matrix( d )
    x_0 = matrix( x_0 )
    y_0 = matrix( y_0 )
    z_0 = matrix( z_0 )
    s_0 = matrix( s_0 )

    ## Initialisation
    iter = 0
    mc   = z_0.size[0]
    X    = matrix( [ x_0.T ] )
    Z    = z_0
    S    = s_0

    ## Initial guess correction
    r_L  = H*x_0 + g - A*y_0 - C*z_0;
    r_A  = b - A.T*x_0;
    r_C  = s_0 + d - C.T*x_0
    r_SZ = S.T*Z

    H_bar = H + C*(spdiag(div(Z,S)))*C.T
    KKT   = matrix( [   [ H_bar, -A.T                                 ],  \
                        [ -A   , matrix( 0.0, (A.size[1],A.size[1]) ) ] ] )
    I   = matrix( range(KKT.size[0]) )  # Identity for LDL

    ## LDL-decomposition
    cvxopt.lapack.sytrf( KKT, I )

    """
    % Affine search direction computation
        r_bar_L   = r_L - C*(S\Z)*(r_C - Z\r_SZ);
        r_A_bar_L = [r_bar_L; r_A];
        tmp(p)    = -L'\(D\(L\(r_A_bar_L(p))));
        dx_aff    = tmp(1:length(r_bar_L))';
        dz_aff    = -(S\Z)*C'*dx_aff + (S\Z)*(r_C - Z\r_SZ);
        ds_aff    = -Z\r_SZ - Z\S*dz_aff;
    """


"""
function [x,y,z,s,data] = interior_point( H, g, A, b, C, d, ...
                                          x0, y0, z0, s0,   ...
                                          eta, tol )

% ======================== Initialization ========================= %
    iter    = 1;
    maxiter = 100;
    m_c     = length(z0);

    % Pre-allocation
    data.x = x0;
% ================================================================= %

% =========== Corrections on z0 and s0 if necessary =============== %
    % Compute residuals and duality gap.
        %Z = bsxfun(@times,z0,eye(m_c));
        %S = bsxfun(@times,s0,eye(m_c));
        Z = diag(z0);
        S = diag(s0);
        e = ones(m_c,1);
        r_L  = H*x0 + g - A*y0 - C*z0;
        r_A  = b - A'*x0;
        r_C  = s0 + d - C'*x0;
        r_SZ = S*Z*e;

    % KKT_bar and LDL-factorizaion.
        H_bar   = H + C*(S\Z)*C';
        KKT_bar = [H_bar, -A; -A', zeros(size(A,2))];
        [L,D,p] = ldl(KKT_bar,'lower','vector');

    % Affine search direction computation
        r_bar_L   = r_L - C*(S\Z)*(r_C - Z\r_SZ);
        r_A_bar_L = [r_bar_L; r_A];
        tmp(p)    = -L'\(D\(L\(r_A_bar_L(p))));
        dx_aff    = tmp(1:length(r_bar_L))';
        dz_aff    = -(S\Z)*C'*dx_aff + (S\Z)*(r_C - Z\r_SZ);
        ds_aff    = -Z\r_SZ - Z\S*dz_aff;

    % Setting initial state of variables
        x = x0;
        y = y0;
        z = max(1,abs(z0+dz_aff));
        s = max(1,abs(s0+ds_aff));
% ================================================================= %

% =========== Residuals and Convergence =========================== %
    % Computation of residuals
        r_L  = H*x + g - A*y - C*z;
        r_A  = b - A'*x;
        r_C  = s + d - C'*x;
        r_SZ = s.*z;
        mu   = sum(r_SZ)/m_c;

    % Check convergence meausure.
        tol_L  = tol*max(1,norm([H,g,A,C],inf));
        tol_A  = tol*max(1,norm([A',b],inf));
        tol_C  = tol*max(1,norm([eye(size(d)),d,C'],inf));
        tol_mu = tol*1e-2*mu;

    % Defining convergence.
        converged = (   ( norm(r_L,inf) <= tol_L  )   && ...
                        ( norm(r_A,inf) <= tol_A  )   && ...
                        ( norm(r_C,inf) <= tol_C  )   && ...
                        (  abs(mu ,inf) <= tol_mu )   );
% ================================================================= %


% ================================================================= %
% Main while loop.
while ~converged
    iter = iter + 1;

    % Precomputing Z and S
        Z = bsxfun(@times,z,eye(m_c));
        S = bsxfun(@times,s,eye(m_c));

    % 1) KKT_bar and LDL-factorizaion.
        H_bar   = H + C*(S\Z)*C';
        KKT_bar = [H_bar, -A; -A', zeros(size(A,2))];
        [L,D,p] = ldl(KKT_bar,'lower','vector');

    % 2) Computation of Affine Step Direction.
        % Residuals.
        r_bar_L   = r_L - C*(S\Z)*(r_C - Z\r_SZ);
        r_A_bar_L = [r_bar_L; r_A];
        tmp(p)    = -L'\(D\(L\(r_A_bar_L(p))));

        dx_aff    = tmp(1:size(H,1))';
        dz_aff    = -(S\Z)*C'*dx_aff + (S\Z)*(r_C - Z\r_SZ);
        ds_aff    = -Z\r_SZ - Z\S*dz_aff;

        % Affine step parameter size determination.
        % Find critical areas
        z_idx = find( dz_aff < 0.0 );
        s_idx = find( ds_aff < 0.0 );

        alpha_z_aff = min([1;-z(z_idx)./(dz_aff(z_idx))]);
        alpha_s_aff = min([1;-s(s_idx)./(ds_aff(s_idx))]);
        alpha_aff   = min(alpha_z_aff,alpha_s_aff);

    % 3) Computation of Duality Gap and Centering Parameter.
        mu_aff = (z + alpha_aff*dz_aff)'*(s + alpha_aff*ds_aff) ...
                  ./m_c;
        sigma  = (mu_aff/mu)^3;

    % 4) Computation of Affine-Centering-Correction Direction
        r_bar_SZ = r_SZ + ds_aff.*dz_aff - sigma*mu;
        r_bar_L   = r_L - C*(S\Z)*(r_C - Z\r_bar_SZ);
        r_A_bar_L = [r_bar_L; r_A];
        tmp(p)    = -L'\(D\(L\(r_A_bar_L(p))));
        dx        = tmp(1:size(H,1))';
        dy        = tmp(size(H,1)+1:end)';
        dz        = -(S\Z)*C'*dx + (S\Z)*(r_C - Z\r_bar_SZ);
        ds        = -Z\r_bar_SZ - Z\S*dz;

        % Step parameter size determination.
        z_idx = find( dz < 0.0 );
        s_idx = find( ds < 0.0 );

        alpha_z   = min([1;-z(z_idx)./(dz(z_idx))]);
        alpha_s   = min([1;-s(s_idx)./(ds(s_idx))]);
        alpha     = min(alpha_z,alpha_s);
        alpha_bar = eta*alpha;

    % 5) Updating step
        x = x + alpha_bar*dx;
        y = y + alpha_bar*dy;
        z = z + alpha_bar*dz;
        s = s + alpha_bar*ds;

    % Recomputing residuals and duality gap));
        r_L  = H*x + g - A*y - C*z;
        r_A  = b - A'*x;
        r_C  = s + d - C'*x;
        r_SZ = s.*z;
        mu   = sum(r_SZ)/m_c;

        converged = (   ( norm(r_L,inf) <= tol_L  ) && ...
                        ( norm(r_A,inf) <= tol_A  ) && ...
                        ( norm(r_C,inf) <= tol_C  ) && ...
                        ( abs(mu)       <= tol_mu ) );

    % Data insertion
        data.x = [data.x, x];

    if ~(iter <= maxiter)
        warning('ERROR!!! Method did not converge.');
        break;
    end
end
"""
