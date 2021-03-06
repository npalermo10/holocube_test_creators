from numpy import *
import numpy as np
import holocube.hc5 as hc5

def get_sph_segs_nbrs(p,n=2):
    ''' get spherical segment angles given the circle radius and the percentage surface area covered. The first segment is centered at 90 degrees (orthogonal).'''
    if p*n >1:
        raise ValueError('you have too many segments or too high percentage visible area')
    ds = []
    hs = []
    d1 = -p
    h1 = -2*d1
    ds.append(d1)
    hs.append(h1)
    for seg in np.arange(n-1):
        d = ds[seg]+hs[seg]
        h = (seg+2)*p
        ds.append(d)
        hs.append(h)
    thetas = []    
    for i_d, d in enumerate(ds):    
        thetas.append(np.array([np.pi/2-np.arcsin((d+hs[i_d])), np.pi/2-np.arcsin(d)]))
    return np.vstack(thetas)[::-1]

def calc_theta_phi(x,y,z, phi_rot = 0):
    '''from cartesian coords return spherical coords,  declination theta (pi - theta), and phi'''
    r = sqrt(x**2 + y**2 + z**2)
    theta = pi - arccos(z/r) # declination
    phi = arctan2(y,x) - phi_rot
    phi[phi > pi] -= 2*pi
    phi[phi < -pi] += 2*pi
    return theta, phi


def rotmat(u=[0.,0.,1.], theta=0.0):
    '''Returns a matrix for rotating an arbitrary amount (theta)
    around an arbitrary axis (u, a unit vector).  '''
    ux, uy, uz = u
    cost, sint = cos(theta), sin(theta)
    uxu = array([[ux*ux, ux*uy, ux*uz],
                 [ux*uy, uy*uy, uz*uy],
                 [ux*uz ,uy*uz , uz*uz]])
    ux = array([[0, -uz, uy],
                [uz, 0, -ux],
                [-uy, ux, 0]])
    return cost*identity(3) + sint*ux + (1 - cost)*uxu

def inds_btw_sph_range(coords_array, theta_min, theta_max, phi_min, phi_max, rx, ry, rz):
    ''' check if coords in range of thetas. return frame and point inds for which wn should be active '''
    rx = -rx
    ry = -ry
    rz = -rz
    x_rot_mat = [[1,0,0],
                [0, cos(rx), -sin(rx)],
                [0, sin(rx), cos(rx)]]
    y_rot_mat = [[cos(ry), 0, sin(ry)],
                 [0 , 1, 0], 
                 [-sin(ry), 0, cos(ry)]]
    z_rot_mat=[[cos(rz), -sin(rz), 0],
               [sin(rz), cos(rz), 0],
               [0 , 0, 1]]

    coords_rotated_x = dot(x_rot_mat, coords_array)
    coords_rotated_yx = dot(y_rot_mat, coords_rotated_x.swapaxes(0, 1))
    coords_rotated_zyx = dot(z_rot_mat, coords_rotated_yx.swapaxes(0, 1))
    xs, ys, zs = coords_rotated_zyx
    theta, phi = calc_theta_phi(xs, ys, zs, phi_rot = 0)
    bool_array = all(array([theta_max >= theta, theta_min <= theta, phi >= phi_min, phi <= phi_max]) , axis = 0)
    return bool_array

class Moving_points():
    '''returns active indexes as dictionary. Key (vel, theta range, phi range)'''
    def __init__(self, numframes,  start_frame, end_frame, dot_density = None, numpoints = 5000, dimensions= [[-4,4],[-2,2],[-30,5]],  default_on = False, vel = 0, direction = [1,0,0], pts_rot = [0,0,0], wn_seq = [], color = 1, annuli = []):

        if dot_density:
            disp_vector = -1* vel * numframes * array(direction)
            window_far = 2 #how far the frustum depth is set. Usually its set to 1
            dimensions = sort(array([[0, disp_vector[0]], [0, disp_vector[1]], [0, disp_vector[2]]]))
            dimensions[:, 0] = dimensions[:, 0] - window_far*2.5
            dimensions[:,1] = dimensions[:, 1] + window_far*2.5
            volume  = abs(product(dimensions[:, 1] - dimensions[:, 0]))
            numpoints = int(dot_density * volume)
            
        self.pts = hc5.stim.Points(hc5.window, numpoints, dims=dimensions, color=color, pt_size=3)
        self.numframes = numframes
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.vel = vel
        self.pts_rx, self.pts_ry, self.pts_rz = pts_rot[0], pts_rot[1], pts_rot[2]
        self.direction = direction
        self.wn_seq = wn_seq
        self.default_on = default_on
        self.annuli = annuli
        
        self.calc_act_inds()
        self.remove_unvisible_points()
        self.get_going_in_out()
        self.get_selector_funcs()
        # self.act_inds = array([where(arr)[0] for arr in self.act_inds]) ## change these into indices instead of boolean arrays for faster running.
        
                    
    def calc_act_inds(self):
        coords_over_t = zeros([self.numframes, 3, self.pts.coords.shape[1]])
        coords_over_t[0] = array([self.pts.coords[0] , self.pts.coords[1], self.pts.coords[2]])
        dist = linalg.norm(self.direction)
        mag  = self.vel/dist
        x_disp = 0
        y_disp = 0
        z_disp = 0
        if len(self.wn_seq) > 0:
            x_disp = self.direction[0] * mag * self.wn_seq
            y_disp = self.direction[1] * mag * self.wn_seq
            z_disp = self.direction[2] * mag * self.wn_seq
            for frame in arange(1, len(self.wn_seq)):
                coords_over_t[frame] = array([coords_over_t[frame-1][0] + x_disp[frame],
                                                coords_over_t[frame-1][1] + y_disp[frame],
                                                coords_over_t[frame-1][2] + z_disp[frame],
                                               ])
        else:
            x_disp = self.direction[0] *mag
            y_disp = self.direction[1] * mag
            z_disp = self.direction[2] * mag
            for frame in arange(1, self.numframes):
                coords_over_t[frame] = array([coords_over_t[frame-1][0] + x_disp,
                                                coords_over_t[frame-1][1] + y_disp,
                                                coords_over_t[frame-1][2] + z_disp,
                                               ])
                coords_over_t[frame] = dot(rotmat([1,0,0], self.pts_rx), coords_over_t[frame]) #rotate about x
                coords_over_t[frame] = dot(rotmat([0,1,0], self.pts_ry), coords_over_t[frame]) #rotate about y
                coords_over_t[frame] = dot(rotmat([0,0,1], self.pts_rz), coords_over_t[frame]) #rotate about z
        act_inds = zeros([coords_over_t.shape[0], coords_over_t.shape[2]])
        if self.default_on:
            act_inds = ones([coords_over_t.shape[0], coords_over_t.shape[2]])
        for annulus in self.annuli:
            if annulus.type_on: 
                act_inds += array(inds_btw_sph_range(coords_over_t, annulus.theta_range[0], annulus.theta_range[1], annulus.phi_range[0], annulus.phi_range[1], annulus.rot[0], annulus.rot[1], annulus.rot[2]))
                         
            else:
                act_inds -= array(inds_btw_sph_range(coords_over_t, annulus.theta_range[0], annulus.theta_range[1], annulus.phi_range[0], annulus.phi_range[1], annulus.rot[0], annulus.rot[1], annulus.rot[2]))

            act_inds = act_inds.astype('bool')
            
        self.act_inds = act_inds
                
    def get_going_in_out(self):
        act_inds_t1 = self.act_inds.astype('int')
        first_frame = zeros([1,self.act_inds.shape[1]])
        act_inds_t0 = vstack([first_frame, self.act_inds[:-1]]).astype('int')
        diff_act_inds_t = act_inds_t1 - act_inds_t0
        inds = array([arange(self.act_inds.shape[1])] * self.act_inds.shape[0])
        self.inds_going_in = [ind[diff_act_inds_t[i_ind] == 1] for i_ind, ind in enumerate(inds)]
        self.inds_going_out = [ind[diff_act_inds_t[i_ind] == -1] for i_ind, ind in enumerate(inds)]
        
    def remove_unvisible_points(self):
        self.act_inds[:self.start_frame] = array([False]*self.act_inds.shape[-1])
        self.act_inds[self.end_frame:] = array([False]*self.act_inds.shape[-1])
        pts_ever_visible = self.act_inds.sum(axis = 0, dtype = 'bool')
        to_remove = array((1 - pts_ever_visible), dtype = 'bool')
        self.pts.remove_subset(to_remove)
        self.act_inds = self.act_inds[:,pts_ever_visible]

    def get_selector_funcs(self):
        self.orig_y = array([self.pts.coords[1, :].copy()]*self.numframes)
        self.orig_x = array([self.pts.coords[0, :].copy()]*self.numframes)
        self.far_y = 20
        self.select_all = array([[1]*self.pts.num] * self.numframes,  dtype='bool')

class Test_creator(object):
    ''' creates annulus experiments. takes Moving_points objects '''
    def __init__(self, numframes, rest_numframes = 300, cl_rest_bar = True):
        self.starts = []
        self.middles = []
        self.ends = []
        self.add_inits()
        self.numframes = numframes
        if cl_rest_bar:
            self.add_rest_bar(numframes = rest_numframes)
        
    def reset(self):
        self.starts = []
        self.middles = []
        self.ends = []
        self.add_inits()
       
    def add_to_starts(self, arr):
        self.starts.append(arr)

    def add_to_middles(self, arr):
        self.middles.append(arr)

    def add_to_ends(self, arr):
        self.ends.append(arr)
        
    def add_inits(self):
        self.add_to_starts([hc5.window.set_bg,  [0.0,0.0,0.0,1.0]])
        ends = [[hc5.window.set_bg,  [0.0,0.0,0.0,1.0]], 
               [hc5.window.set_ref, 0, [0,0,0]],
               [hc5.window.set_ref, 1, [0,0,0]],
               [hc5.window.set_ref, 2, [0,0,0]],
               [hc5.window.set_ref, 3, [0,0,0]],
               [hc5.window.reset_pos]]
        for end in ends:                   
            self.add_to_ends(end)

    def add_rest_bar(self, numframes):        
        rbar = hc5.stim.cbarr_class(hc5.window, dist=1)
        starts =  [[hc5.window.set_far,         2],
                   [hc5.window.set_bg,          [0.0,0.0,0.0,1.0]],
                   [hc5.arduino.set_lmr_scale,  -.1],
                   [rbar.set_ry,               0],
                   [rbar.switch,               True] ]
        middles = [[rbar.inc_ry,               hc5.arduino.lmr]]
        ends =    [[rbar.switch,               False],
                   [hc5.window.set_far,         2]]
        hc5.scheduler.add_rest(numframes, starts, middles, ends)

    def add_static_bar(self, ry_pos = 0, start_t = 0, end_t = 1):        
        rbar = hc5.stim.cbarr_class(hc5.window, dist=1)
        state = array([0.0] * self.numframes)
        state[int(self.numframes*start_t): int(self.numframes *end_t)] = 1
        self.add_to_starts([rbar.set_ry,               ry_pos])
        self.add_to_middles([rbar.on,               state])
        self.add_to_ends([rbar.switch,               False])
        
    def add_index_lights(self, ref_light, index, num_mod = 1):
        '''adds index light sequence to middles. turns off light at end'''
        index += num_mod
        seq = hc5.tools.test_num_flash(index, self.numframes)
        self.add_to_starts([hc5.window.set_ref, ref_light, (0,0,0)])
        self.add_to_middles([hc5.window.set_ref, ref_light, seq])
        self.add_to_ends([hc5.window.set_ref, ref_light, (0,0,0)])

    def add_wn_lights(self, ref_light, ms):
        wn_seq = array([(0,175+wn*80,0) for wn in ms], dtype='int')
        wn_seq[-1] = array([0, 0, 0])
        self.add_to_starts([hc5.window.set_ref, ref_light, (0,0,0)])
        self.add_to_middles([hc5.window.set_ref, ref_light, wn_seq])
        self.add_to_ends([hc5.window.set_ref, ref_light, (0,0,0)])

        
    def add_simple_pt_field(self, density = 30, trans_v = [0.00, 0.00, 0.00], rot_v = [0.00, 0.00,0.00], start_t = 0, end_t = 1, **kwargs):
        ''' simple translating point field'''
        trans_v = array(trans_v)
        if trans_v.shape[1] == 1:
            trans_v.repeat(numframes).reshape(3, numframes)
        dx, dy, dz = trans_v
        rx, ry, rz = rot_v
        dimensions = array([[cumsum(dx).min(), cumsum(dx).max()],
                               [cumsum(dy).min(), cumsum(dy).max()],
                               [cumsum(dz).min(), cumsum(dz).max()]])
        window_far = 2 #how far the frustum depth is set. Usually its set to 1
        dimensions[:, 0] = dimensions[:, 0] - window_far*2.5
        dimensions[:,1] = dimensions[:, 1] + window_far*2.5
        volume  = abs(product(dimensions[:, 1] - dimensions[:, 0]))
        numpoints = int(density * volume)
       
        pts = hc5.stim.Points(hc5.window, num = numpoints, dims = dimensions, **kwargs)
        state = array([0.0] * self.numframes)
        state[int(self.numframes*start_t): int(self.numframes *end_t)] = 1
        self.add_to_middles([pts.on, state])
        self.add_to_middles([pts.inc_px, dx])
        self.add_to_middles([pts.inc_py, dy])
        self.add_to_middles([pts.inc_pz, dz])
        self.add_to_middles([pts.inc_rx, rx])
        self.add_to_middles([pts.inc_ry, ry])
        self.add_to_middles([pts.inc_rz, rz])
        self.add_to_ends([pts.switch, False])
        self.add_to_ends([pts.inc_px, -dx.sum()])
        self.add_to_ends([pts.inc_py, -dy.sum()])
        self.add_to_ends([pts.inc_pz, -dz.sum()])
        self.add_to_ends([pts.inc_rx, -rx*self.numframes])
        self.add_to_ends([pts.inc_ry, -ry*self.numframes])
        self.add_to_ends([pts.inc_rz, -rz*self.numframes])
        
        
    def add_simple_trans_pt_field(self, density = 30, dx = 0, dy = 0, dz = 0.01, start_t = 0, end_t = 1, **kwargs):
        ''' simple translating point field'''
        vel = linalg.norm([dx, dy, dz])
        disp_vector = -1* vel * self.numframes * array([dx, dy, dz])
        window_far = 2 #how far the frustum depth is set. Usually its set to 1
        dimensions = sort(array([[0, disp_vector[0]], [0, disp_vector[1]], [0, disp_vector[2]]]))
        dimensions[:, 0] = dimensions[:, 0] - window_far*2.5
        dimensions[:,1] = dimensions[:, 1] + window_far*2.5
        volume  = abs(product(dimensions[:, 1] - dimensions[:, 0]))
        numpoints = int(density * volume)
       
        pts = hc5.stim.Points(hc5.window, num = numpoints, dims = dimensions, **kwargs)
        state = array([0.0] * self.numframes)
        state[int(self.numframes*start_t): int(self.numframes *end_t)] = 1
        self.add_to_middles([pts.on, state])
        self.add_to_middles([pts.inc_px, dx])
        self.add_to_middles([pts.inc_py, dy])
        self.add_to_middles([pts.inc_pz, dz])
        self.add_to_ends([pts.switch, False])
        self.add_to_ends([pts.inc_px, -dx*self.numframes])
        self.add_to_ends([pts.inc_py, -dy*self.numframes])
        self.add_to_ends([pts.inc_pz, -dz*self.numframes])
       
    def add_to_scheduler(self, and_reset = True):
        ''' send test to scheduler and reset test if flag is True.'''
        hc5.scheduler.add_test(self.numframes, self.starts, self.middles, self.ends)
        if and_reset:
            self.reset()
            
    def add_bg_static(self, intensity= 1.0 , start_t = 0.0, end_t = 1.0):
        ''' add a static bg color from a start to end time'''
        bg_color = (intensity, intensity, intensity, 1.0)
        bg_state = array([(0.0,0.0,0.0,1.0)] * self.numframes)
        bg_state[int(self.numframes*start_t): int(self.numframes *end_t)] = bg_color
        self.add_to_middles([hc5.window.set_bg, bg_state])

    def close_loop_lmr(self, lmr_scale = 0.0017):
        self.add_to_middles([hc5.arduino.set_lmr_scale,   lmr_scale])

    def save_ry(self, index_val):    
        self.add_to_middles([hc.window.record, index_val, arange(self.numframes, dtype='int'), hc5.window.get_rot])

    def add_sph_seg(self, rx = 0, ry=0, rz = 0, **kwargs):
        ss = hc5.stim.Spherical_segment(hc5.window, **kwargs)
        self.add_to_starts([ss.set_rx,                  rx])
        self.add_to_starts([ss.set_ry,                  ry])
        self.add_to_starts([ss.set_rz,                  rz])
        self.add_to_starts([ss.switch,                  True])
        self.add_to_ends([ss.switch,                  False])    

    def add_looming_stim(self, rx = 0, ry = 0, rz =0 ,start_t = 0 , end_t= 1, **kwargs):
        ## rotation happens in rx, ry, rz order
        disk = hc5.stim.disk_class(hc5.window, radius = 0.2, **kwargs)
        disk.set_rx = rx
        disk.set_ry = ry
        disk.set_rz = rz
        length = linalg.norm(disk.pos)
        v = disk.pos/length
        
        self.add_to_starts([disk.on, 1])

        
    def add_moving_points(self, start_t= 0 , end_t = 1, start_fr = [], end_fr = [], **kwargs):
        image_state = array([False] * self.numframes)
        image_state[int(self.numframes*start_t): int(self.numframes*end_t)] = True
        if not start_fr:
            start_fr = int(self.numframes*start_t)
        if not end_fr:    
            end_fr = int(self.numframes*end_t)
        
        points = Moving_points(numframes = self.numframes, start_frame= start_fr, end_frame = end_fr, **kwargs)

        dx = 0
        dy = 0
        dz = 0
        if 'wn_seq' in kwargs:
            wn_seq = kwargs['wn_seq']
            dx, dy, dz = zeros(self.numframes), zeros(self.numframes), zeros(self.numframes)
            dx[start_fr:start_fr + len(wn_seq)] = (points.direction[0] * points.vel * wn_seq/linalg.norm(points.direction))[:end_fr-start_fr]
            dy[start_fr:start_fr + len(wn_seq)] = (points.direction[1] * points.vel * wn_seq/linalg.norm(points.direction))[:end_fr - start_fr]
            dz[start_fr:start_fr + len(wn_seq)] = (points.direction[2] * points.vel * wn_seq/linalg.norm(points.direction))[:end_fr - start_fr]
            self.add_to_ends([points.pts.inc_px, -points.direction[0] * points.vel * sum(dx)/linalg.norm(points.direction)])
            self.add_to_ends([points.pts.inc_py, -points.direction[1] * points.vel * sum(dy)/linalg.norm(points.direction)])
            self.add_to_ends([points.pts.inc_pz, -points.direction[2] * points.vel * sum(dz)/linalg.norm(points.direction)]) 
        
        else:
            dx = points.direction[0] * points.vel/linalg.norm(points.direction)
            dy = points.direction[1] * points.vel/linalg.norm(points.direction)
            dz = points.direction[2] * points.vel/linalg.norm(points.direction)
                      
            self.add_to_ends([points.pts.inc_px, -points.direction[0] * points.vel/linalg.norm(points.direction)*points.act_inds.shape[0]])
            self.add_to_ends([points.pts.inc_py, -points.direction[1] * points.vel/linalg.norm(points.direction)*points.act_inds.shape[0]])
            self.add_to_ends([points.pts.inc_pz, -points.direction[2] * points.vel/linalg.norm(points.direction)*points.act_inds.shape[0]]) 
            self.add_to_ends([points.pts.inc_rx, -points.pts_rx*180/pi*points.act_inds.shape[0]])
            self.add_to_ends([points.pts.inc_ry, -points.pts_ry*180/pi*points.act_inds.shape[0]])
            self.add_to_ends([points.pts.inc_rz, -points.pts_rz*180/pi*points.act_inds.shape[0]])

        for annulus in points.annuli:
                if annulus.bg_color is not None:
                    self.add_sph_seg(rx = annulus.rot[0], ry = annulus.rot[1], rz = annulus.rot[2], color = annulus.bg_color, polang_top = annulus.theta_range[0]*360/(2*pi), polang_bot = annulus.theta_range[1]*360/(2*pi))
        self.add_to_starts([points.pts.subset_inc_py, points.select_all[0], points.far_y])
        self.add_to_ends([points.pts.subset_set_py, points.select_all[0], points.orig_y[0]])
        self.add_to_middles([points.pts.on, image_state])
        self.add_to_middles([points.pts.subset_inc_py, points.inds_going_out, points.far_y])
        self.add_to_middles([points.pts.subset_inc_py, points.inds_going_in, -points.far_y])
        self.add_to_middles([points.pts.inc_px,    dx])
        self.add_to_middles([points.pts.inc_py,    dy])
        self.add_to_middles([points.pts.inc_pz,    dz])
        self.add_to_middles([points.pts.inc_rx,    points.pts_rx*180/pi])
        self.add_to_middles([points.pts.inc_ry,    points.pts_ry*180/pi])
        self.add_to_middles([points.pts.inc_rz,    points.pts_rz*180/pi])
        self.add_to_ends([points.pts.on, 0])
    
class Ann_test_creator(Test_creator):
    ''' creates annulus experiments. takes Moving_points objects. This should be phased out soon 5/29/2019. '''
    def __init__(self, num_frames):
        Test_creator.__init__(self, num_frames)
        
class Annulus():
    def __init__(self, theta_range  = [0, pi], phi_range = [-pi, pi], rot = [0,0,0], type_on = True, bg_color = None):
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.rot = rot
        self.type_on = type_on
        self.bg_color = bg_color
    
class Motion_ill_test_creator(Test_creator):
    '''create static image motion illusions. pick image location, onset time, end time'''
    def __init__(self, num_frames):
        Test_creator.__init__(self, num_frames)

    def add_image(self, img, left_edge=-pi/4, right_edge=pi/4, bottom_edge=-pi/4, top_edge=pi/4, rx = 0, ry = 0, rz = 0, start_t = 0.0, end_t = 1.0, x_inv = False):
        if x_inv:
            left_edge = -left_edge
            right_edge = -right_edge
        image = hc5.stim.Quad_image(hc5.window, left= left_edge, right= right_edge, bottom= bottom_edge, top= top_edge, xdivs=24, ydivs=1, dist=1.5)
        image.load_image(img)
        image_state = array([False] * self.numframes)
        image_state[int(self.numframes*start_t): int(self.numframes*end_t)] = True
        self.add_to_starts([hc5.window.set_far,          3])
        self.add_to_ends([hc5.window.set_far,          2])
        self.add_to_starts([image.set_rx, rx])
        self.add_to_starts([image.set_ry, ry])
        self.add_to_starts([image.set_rz, rz])
        self.add_to_middles([image.on, image_state])
        self.add_to_ends([image.on, False])

class Grating_test_creator(Test_creator):
    def __init__(self, num_frames):
        Test_creator.__init__(self, num_frames)
        self.set_perspective()
        
    def set_perspective(self):
        self.add_to_starts([hc5.window.set_viewport_projection, 0, 0])
        self.add_to_ends([hc5.window.set_viewport_projection, 0, 1])

    def reset(self):
        super(Grating_test_creator, self).reset()
        self.set_perspective()
        
    def add_plaid(self, start_t = 0, end_t = 1, viewport = 0, static = False,  **kwargs):
        sp = hc5.stim.grating_class(hc5.window, vp =viewport, fast=True)
        if static:
            sp.add_plaid(tf1 = 0.0001, tf2= 0.0001, **kwargs)
        else:
            sp.add_plaid(**kwargs)
        state = array([0.0] * self.numframes)
        state[int(self.numframes*start_t): int(self.numframes *end_t)] = 1
        self.add_to_starts([sp.choose_grating, 0])
        self.add_to_middles([sp.on, state])
        self.add_to_middles([sp.animate, arange(self.numframes)])
        self.add_to_ends([sp.on, 0])

    def add_grating(self, start_t = 0, end_t = 1, viewport = 0, static = False, **kwargs):
        sp = hc5.stim.grating_class(hc5.window, vp =viewport, fast=True)
        if static:
            sp.add_grating(tf = 0.0001, **kwargs)
        else:
            sp.add_grating(**kwargs)
        state = array([0.0] * self.numframes)
        state[int(self.numframes*start_t): int(self.numframes *end_t)] = 1
        self.add_to_starts([sp.choose_grating, 0])
        self.add_to_middles([sp.on, state])
        self.add_to_middles([sp.animate, arange(self.numframes)])
        self.add_to_ends([sp.on, 0])
        

class Condition:
    ''' use this instance for a list of factors that will be cycled'''
    def __init__(self, factors, flashes_to_add= 1):
        self.factors = factors
        self.flashes_to_add = flashes_to_add
        self.curr_index = 0


class Experiment:
    '''contains test and conditions for setting up multifactorial experiments'''
    def __init__(self, numframes, restnumframes = 300):
        hc.scheduler.add_exp()
        self.numframes = numframes
        self.restnumframes =restnumframes
        self.test = False
        self.conditions = []
        self.light_mod = 1
        self.add_rest_bar()

    def add_condition(self, factors):
        self.conditions.append(Condition(factors), self.light_mod)
        self.light_mod += 1
        
    def add_test(self, test):
        self.test = test
        curr_light = 0
        for i_condition, condition in enumerate(conditions):
            self.test.add_index_lights(0, i_condition + condition.flashes_to_add)
                

    def add_rest_bar(self):
        rbar = hc5.stim.cbarr_class(hc5.window, dist=1)
        starts =  [[hc5.window.set_far,         2],
                   [hc5.window.set_bg,          [0.0,0.0,0.0,1.0]],
                   [hc5.arduino.set_lmr_scale,  -.1],
                   [rbar.set_ry,               0],
                   [rbar.switch,               True] ]
        middles = [[rbar.inc_ry,               hc5.arduino.lmr]]
        ends =    [[rbar.switch,               False],
                   [hc5.window.set_far,         2]]
        hc5.scheduler.add_rest(self.restnumframes, starts, middles, ends)
    

