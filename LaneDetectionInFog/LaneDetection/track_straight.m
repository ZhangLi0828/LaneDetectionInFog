function [agl_l,agl_r,r1,r2] = track(frame,j,agl_l,agl_r,r1,r2)
[frame,area] = preprocess(frame);
[nonzeroy,nonzerox] = find(area~=0);
margin = 5;
map = zeros(size(frame));

% ÔÚmargin·¶Î§ÄÚËÑË÷
left_inds =  (( nonzerox > (  (r1 - nonzeroy*sind(agl_l)) /cosd(agl_l) - margin ) )... 
                    & ( nonzerox < (  (r1 - nonzeroy*sind(agl_l)) /cosd(agl_l) + margin ) ) );

right_inds =  (( nonzerox > (  (r2 - nonzeroy*sind(agl_r)) /cosd(agl_r) - margin ) )... 
                    & ( nonzerox < (  (r2 - nonzeroy*sind(agl_r)) /cosd(agl_r) + margin ) ) );
                
leftx = nonzerox(left_inds);
lefty = nonzeroy(left_inds);
rightx = nonzerox(right_inds);
righty = nonzeroy(right_inds);

x = [leftx;rightx];
y = [lefty;righty];

for m = 1:size(x)
        map(y(m),x(m))=1;
end

[agl_l,agl_r,r1,r2] = LaneDetect(frame,j,map);
end
