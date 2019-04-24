from numpy import *
import holocube.hc5 as hc5

def calc_theta_phi(x,y,z, phi_rot = 0):
    '''from cartesian coords return spherical coords,  declination theta (pi - theta), and phi'''
    r = sqrt(x**2 + y**2 + z**2)
    theta = pi - arccos(z/r) # declination
    phi = arctan2(y,x) - phi_rot
    phi[phi > pi] -= 2*pi
    phi[phi < -pi] += 2*pi
    return theta, phi

def inds_btw_sph_range(coords_array, theta_min, theta_max, phi_min, phi_max):
    ''' check if coords in range of thetas. return frame and point inds for which wn should be active '''
    theta, phi = calc_theta_phi(coords_array[:,0], coords_array[:,1], coords_array[:,2], phi_rot = 0)
    bool_array = all(array([theta_max >= theta, theta_min <= theta, phi >= phi_min, phi <= phi_max]) , axis = 0)
    return bool_array

class Moving_points():
    '''returns active indexes as dictionary. Key (vel, theta range, phi range)'''
    def __init__(self, numframes,  start_frame, end_frame, dot_density = None, numpoints = 5000, dimensions= [[-4,4],[-2,2],[-30,5]],  vel = 0, direction = [1,0,0], theta_ranges  = [[0, pi]], phi_ranges = [[-pi, pi]], rx = 0, ry = 0, rz = 0, wn_seq = [], color = 0.5):

        if dot_density:
            disp_vector = -1* vel * numframes * array(direction)
            window_far = 2 #how far the frustum depth is set. Usually its set to 1
            dimensions = sort(array([[0, disp_vector[0]], [0, disp_vector[1]], [0, disp_vector[2]]]))
            dimensions[:, 0] = dimensions[:, 0] - window_far
            dimensions[:,1] = dimensions[:, 1] + window_far
            volume  = abs(product(dimensions[:, 1] - dimensions[:, 0]))
            numpoints = int(dot_density * volume)
            
        self.pts = hc5.stim.Points(hc5.window, numpoints, dims=dimensions, color=color, pt_size=3)
        self.numframes = numframes
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.vel = vel
        self.direction = direction
        self.theta_ranges = array(theta_ranges)
        self.phi_ranges = array(phi_ranges)
        if self.theta_ranges.shape != self.phi_ranges.shape:
            if self.theta_ranges.shape[0] == 1:
                self.theta_ranges = array([self.theta_ranges[0]] * self.phi_ranges.shape[0])
            if self.phi_ranges.shape[0] == 1:
                self.phi_ranges = array([self.phi_ranges[0]] * self.theta_ranges.shape[0])
        self.wn_seq = wn_seq

        self.act_inds = []        
        self.calc_act_inds()
        self.remove_unvisible_points()
        self.get_selector_funcs()
        self.act_inds = array([where(arr)[0] for arr in self.act_inds]) ## change these into indices instead of boolean arrays for faster running.
        
    def calc_act_inds(self):
        for i_theta_range, theta_range in enumerate(self.theta_ranges):
            for i_phi_range, phi_range in enumerate(self.phi_ranges):
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
                    for frame in arange(1, self.numframes):
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
                self.act_inds.append(array(inds_btw_sph_range(coords_over_t, theta_range[0], theta_range[1], phi_range[0], phi_range[1])))
        self.act_inds = array(self.act_inds)
        self.act_inds = self.act_inds.sum(axis=0, dtype = 'bool')
         
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
        self.far_y = array([[10] * self.pts.num] * self.numframes)
        self.select_all = array([[1]*self.pts.num] * self.numframes,  dtype='bool')

class Test_creator():
    ''' creates annulus experiments. takes Moving_points objects '''
    def __init__(self, numframes):
        self.starts = []
        self.middles = []
        self.ends = []
        self.add_inits()
        self.numframes = numframes
        
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
        
    def add_index_lights(self, ref_light, index, num_mod = 0):
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

class Ann_test_creator(Test_creator):
    ''' creates annulus experiments. takes Moving_points objects '''
    def __init__(self, num_frames):
        Test_creator.__init__(self, num_frames)
    
    def add_annulus(self, start_t= 0 , end_t = 1, **kwargs):
        image_state = array([False] * self.numframes)
        image_state[int(self.numframes*start_t): int(self.numframes*end_t)] = True
        end_fr = int(self.numframes*end_t)
        start_fr = int(self.numframes*start_t)
        annulus = Moving_points(numframes = self.numframes, start_frame= start_fr, end_frame = end_fr,   **kwargs)

        dx = 0
        dy = 0
        dz = 0
        if 'wn_seq' in kwargs:
            wn_seq = kwargs['wn_seq']
            dx, dy, dz = zeros(self.numframes), zeros(self.numframes), zeros(self.numframes)
            dx[start_fr:] = (annulus.direction[0] * annulus.vel * wn_seq/linalg.norm(annulus.direction))[:end_fr-start_fr]
            dy[start_fr:] = (annulus.direction[1] * annulus.vel * wn_seq/linalg.norm(annulus.direction))[:end_fr - start_fr]
            dz[start_fr:] = (annulus.direction[2] * annulus.vel * wn_seq/linalg.norm(annulus.direction))[:end_fr - start_fr]
            self.add_to_ends([annulus.pts.inc_px, -annulus.direction[0] * annulus.vel * sum(wn_seq[:end_fr])/linalg.norm(annulus.direction)])
            self.add_to_ends([annulus.pts.inc_py, -annulus.direction[1] * annulus.vel * sum(wn_seq[:end_fr])/linalg.norm(annulus.direction)])
            self.add_to_ends([annulus.pts.inc_pz, -annulus.direction[2] * annulus.vel * sum(wn_seq[:end_fr])/linalg.norm(annulus.direction)]) 
        
        else:
            dx = annulus.direction[0] * annulus.vel/linalg.norm(annulus.direction)
            dy = annulus.direction[1] * annulus.vel/linalg.norm(annulus.direction)
            dz = annulus.direction[2] * annulus.vel/linalg.norm(annulus.direction)
            self.add_to_ends([annulus.pts.inc_px, -annulus.direction[0] * annulus.vel/linalg.norm(annulus.direction)*annulus.act_inds.shape[0]])
            self.add_to_ends([annulus.pts.inc_py, -annulus.direction[1] * annulus.vel/linalg.norm(annulus.direction)*annulus.act_inds.shape[0]])
            self.add_to_ends([annulus.pts.inc_pz, -annulus.direction[2] * annulus.vel/linalg.norm(annulus.direction)*annulus.act_inds.shape[0]]) 
        

        self.add_to_middles([annulus.pts.on, image_state])
        self.add_to_middles([annulus.pts.subset_set_py, annulus.select_all, annulus.far_y])
        self.add_to_middles([annulus.pts.subset_set_py, annulus.act_inds, annulus.orig_y])
        self.add_to_middles([annulus.pts.inc_px,    dx])
        self.add_to_middles([annulus.pts.inc_py,    dy])
        self.add_to_middles([annulus.pts.inc_pz,    dz])
        self.add_to_ends([annulus.pts.on, 0])

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

    def add_bg_static(self, intensity= 1.0 , start_t = 0.0, end_t = 1.0):
        ''' add a static bg color from a start to end time'''
        bg_color = (255*intensity, 255*intensity, 255*intensity, 1.0)
        bg_state = array([(0.0,0.0,0.0,1.0)] * self.numframes)
        bg_state[int(self.numframes*start_t): int(self.numframes *end_t)] = bg_color
        self.add_to_middles([hc5.window.set_bg, bg_state])

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
    

