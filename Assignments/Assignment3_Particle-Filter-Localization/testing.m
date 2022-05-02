close all
clc

load occmap.mat;
figure
pcolor(ogp);
colormap(1-gray);
shading('flat');

row_particle = 60;
col_particle = 150;
hold on
r = 0.15/ogres;
th = 0;
% set(plot([col_particle col_particle+r*cos(th)]', [row_particle row_particle+r*sin(th)]', 'k-'),'LineWidth',2);
set(plot( col_particle, row_particle, 'g.' ),'MarkerSize',20);

ogres = 0.05;
theta = -pi;
beam_angle = 0;
thresh = 0.5;
y = exp_meas(row_particle, col_particle, theta, beam_angle, ogp, ogres, thresh, 5)

function y_exp = exp_meas(row, col, theta, beam_angle, map, ogres, thresh, y_max)
    
    new_angle = atan2(sin(theta+beam_angle), cos(theta+beam_angle));
    incr  = 0;
    r_p = row;
    c_p = col;
    
    % angle is between -pi/4 and pi/4
    if -pi/4<=new_angle && new_angle<=pi/4   
        while r_p>0 && c_p>0 && r_p<=180 && c_p<=300 && map(r_p, c_p)<thresh
            incr = incr + 1;
            c_p = col + incr;
            r_p = row + round(incr*tan(new_angle));

        end        
        y_exp = abs((incr/cos(new_angle))*ogres);
        
    % angle is between -3pi/4 and 3pi/4    
    elseif 3*pi/4<=new_angle || new_angle<=-3*pi/4
        while r_p>0 && c_p>0 && r_p<=180 && c_p<=300 && map(r_p, c_p)<thresh
            incr = incr + 1;
            c_p = col - incr;
            r_p = row - round(incr*tan(new_angle));

        end        
        y_exp = abs((incr/cos(new_angle))*ogres);
        
        
    % angle is between pi/4 and 3pi/4
    elseif pi/4<new_angle && new_angle<3*pi/4 
        while r_p>0 && c_p>0 && r_p<=180 && c_p<=300 && map(r_p, c_p)<thresh
            incr = incr + 1;
            r_p = row + incr;
            c_p = col + round(incr/tan(new_angle));
        end        
        y_exp = abs((incr/sin(new_angle))*ogres);
        
    
    % angle is between -pi/4 and -3pi/4    
    else
        while r_p>0 && c_p>0 && r_p<=180 && c_p<=300 && map(r_p, c_p)<thresh
            incr = incr + 1;
            r_p = row - incr;
            c_p = col - round(incr/tan(new_angle));
        end        
        y_exp = abs((incr/sin(new_angle))*ogres);
        
    
    end
    
end


