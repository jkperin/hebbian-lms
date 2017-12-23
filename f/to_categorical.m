function d = to_categorical(y, n)
d = zeros(n, length(y));
for k = 1:length(y)
    d(y(k)+1, k) = 1;
end