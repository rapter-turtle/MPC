###	Third Order Low Pass Filter y = C(s)*u
# low pass filter C(s) = (3*omega_cutoff^2*s + omega_cutoff^3)/(s^3 + 3*omega_cutoff*s^2 + 3*omega_cutoff^2*s + omega_cutoff^3)

# # first find derivative of input signal (i.e. u = track_error, u_dot = d/dt(track_error) )
# self.u_dot = 1.0/self.dt*(track_error - self.u) # u_dot = 1/dt*(u - u_old)
# self.u = track_error # set current u to track_error (in next iteration, this is automatically u_old)

# self.y_ddot = self.y_ddot + self.dt*(-3*self.omega_cutoff.dot(self.y_ddot) - 3*(self.omega_cutoff**2).dot(self.y_dot) - (self.omega_cutoff**3).dot(self.y) + 3*(self.omega_cutoff**2).dot(self.u_dot) + (self.omega_cutoff**3).dot(self.u) )
# self.y_dot = self.y_dot + self.dt*(self.y_ddot)
# self.y = self.y + self.dt*(self.y_dot)

# low pass filter output is L1 desired
# self.x_L1_des = self.y