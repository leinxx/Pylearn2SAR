function result = verify_vector(im,vec)
    
    samples = zeros(size(vec,1),1);
    for i = 1:size(vec,1)
        samples(i) = im(ceil(vec(i,2)),ceil(vec(i,1)));
    end
    result = [samples, vec(:,3)];
    
end