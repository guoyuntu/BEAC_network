function [features]= encoding_video_recognition(centers, videofeat,opt)
   
    bow = bowcal(videofeat,centers,opt);
    features = bow/size(videofeat,1);

end

function bow = bowcal(fea,centers,opt)

    bow = zeros(1,size(centers,1));
    %size(fea)
    %fea = reshape(fea, 30, 4096);
    m = size(fea,1);
     
    simi = fea*centers';
    
    for wi = 1:m
        weight = linspace(1,0,opt);
        for d = 1:length(weight)
            [Y, I] = max(simi(wi,:));
            bow(1,I) = bow(1,I) + Y * weight(d);
            simi(wi,I) = 0;
        end
    end
    
end