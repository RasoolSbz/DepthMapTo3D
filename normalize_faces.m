switch method
    case 'linear'
        for i = 1 : length(faces)
            normalFaces{i} = faces{i};
            %            normalFaces{i}.shape = (faces{i}.shape - min(faces{i}.shape(:))) * (xmax- xmin) / (max(faces{i}.shape(:)) ...
            %                - min(faces{i}.shape(:))) + xmin;
            %            normalFaces{i}.texture = faces{i}. texture ./ max(faces{i}.texture(:));
            normalFaces{i}.shape = (faces{i}.shape - ...
                repmat([min(faces{i}.shape(: , 1)) , ...
                min(faces{i}.shape(: , 2)), min(faces{i}.shape(: , 3))] , size(faces{i}.shape, 1) , 1))...
                * (xmax - xmin) ./ (repmat([max(faces{i}.shape(: , 1))...
                , max(faces{i}.shape(: , 2)), max(faces{i}.shape(: , 3))]...
                -[min(faces{i}.shape(: , 1))...
                , min(faces{i}.shape(: , 2)), min(faces{i}.shape(: , 3))] , size(faces{i}.shape, 1) , 1))...              
                +xmin;%;(max(faces{i}.shape(:))- min(faces{i}.shape(:))) + xmin;
            
%             normalFaces{i}.landmarks_shape_3d = (faces{i}.landmarks_shape_3d' - ...
%                 repmat([min(faces{i}.shape(: , 1)) , ...
%                 min(faces{i}.shape(: , 2)), min(faces{i}.shape(: , 3))] , size(faces{i}.landmarks_shape_3d', 1) , 1))...
%                 * (xmax - xmin) ./ (repmat([max(faces{i}.shape(: , 1))...
%                 , max(faces{i}.shape(: , 2)), max(faces{i}.shape(: , 3))]...
%                 -[min(faces{i}.shape(: , 1))...
%                 , min(faces{i}.shape(: , 2)), min(faces{i}.shape(: , 3))] , size(faces{i}.landmarks_shape_3d', 1) , 1))...              
%                 +xmin;%;(max(faces{i}.shape(:))- min(faces{i}.shape(:))) + xmin;
            
%             normalFaces{i}.texture = faces{i}. texture ./ max(faces{i}.texture(:));
        end
        
end