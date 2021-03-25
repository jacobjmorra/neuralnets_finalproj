function generate_images(rng, num) 

%INPUT rng = numerosity of images to generate, num = number of each 
A = zeros(32);

for ii = 1:rng %number of types of images
    
    for jj = 1:num %number of each image
        
        for kk = 1:rng % how many entries to change to 1 
            A(randsample(32, 1), randsample(32,1)) = 1 ;
        end
        
        filename = strcat('image_', num2str(ii),'_', num2str(jj),'.png');
        imwrite(A, filename) 
    end
end
