function [Xout, Yout] = cut_tail_with_equal_values(X, Y)
% Given to vectors X, Y of equal size, representing a curve,
% it removes the tail of the curve that is completely flat 
% (i.e. it removes all the identical Y values)
%

assert(numel(X) == numel(Y));

Xout = X;
Yout = Y;

for i=numel(Y):-1:2
  if Y(i) ~= Y(i-1)
    Xout = X(1:i);
    Yout = Y(1:i);
    return;
  end
end

end
