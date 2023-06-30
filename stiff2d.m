tic
syms Ex Ey nuxy Gxy

S = [1/Ex, -nuxy/Ex, 0;
    -nuxy/Ex, 1/Ey, 0;
    0, 0, 1/Gxy];
C = inv(S);

syms theta
T = [cos(theta)^2, sin(theta)^2, -2*cos(theta)*sin(theta);
    sin(theta)^2, cos(theta)^2, 2*cos(theta)*sin(theta);
    cos(theta)*sin(theta), -cos(theta)*sin(theta), cos(theta)^2-sin(theta)^2];
Cr = T * C * transpose(T);

syms r s
a = 1/4 * (1-r) * (1-s);
b = 1/4 * (1+r) * (1-s);
c = 1/4 * (1+r) * (1+s);
d = 1/4 * (1-r) * (1+s);

B1 = [diff(a,r), 0; 0, diff(a,s); diff(a,s), diff(a,r)];
B2 = [diff(b,r), 0; 0, diff(b,s); diff(b,s), diff(b,r)];
B3 = [diff(c,r), 0; 0, diff(c,s); diff(c,s), diff(c,r)];
B4 = [diff(d,r), 0; 0, diff(d,s); diff(d,s), diff(d,r)];

B = [B1, B2, B3, B4];

BCB = transpose(B) * Cr * B;
K = int(int(BCB, r, -1, 1), s, -1, 1);
K = simplify(K);
dK = diff(K, theta);
dK = simplify(dK);

f = fopen('dkdt2d.py','w');
fprintf(f,'import numpy as np\n');
fprintf(f,'def dkdt2d(Ex,Ey,nuxy,nuyz,Gxy,T,V):\n');
fprintf(f,'    c2 = np.cos(2*T)\n');
fprintf(f,'    c4 = np.cos(4*T)\n');
fprintf(f,'    s2 = np.sin(2*T)\n');
fprintf(f,'    s4 = np.sin(4*T)\n');
fprintf(f,'    dkdt = np.zeros((8,8))\n');
for i = 1:8
    for j = i:8
        line = string(dK(i,j));
        line = strrep(line,'^','**');
        line = strrep(line,'cos(2*theta)','c2');
        line = strrep(line,'cos(4*theta)','c4');
        line = strrep(line,'sin(2*theta)','s2');
        line = strrep(line,'sin(4*theta)','s4');
        
        % searches if expression was already calculated in previous terms
        found = false;
        for k = 1:i
            if found
                break
            end
            for l = k:8
                if i==k && j==l
                    break
                end
                if dK(i,j) == dK(k,l)
                    line = sprintf('dkdt[%d][%d]', k-1, l-1);
                    found = true;
                    break
                end
                if dK(i,j) == -dK(k,l)
                    line = sprintf('-dkdt[%d][%d]', k-1, l-1);
                    found = true;
                    break
                end
            end
        end
        fprintf(f,'    dkdt[%d][%d] = %s\n', i-1, j-1, line);
    end
end
fprintf(f,'    dkdt += dkdt.T - np.diag(dkdt.diagonal())\n');
fprintf(f,'    dkdt *= V\n');
fprintf(f,'    return dkdt');
fclose(f);
toc