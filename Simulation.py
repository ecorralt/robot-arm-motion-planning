import matplotlib.pyplot as plt
import jax.numpy as jnp
from Kinematics import forward_kinematics , newton_raphson 
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from matplotlib.widgets import Slider
from trajectory_optimizer import optimizer, SHOULDER_LIMITS, ELBOW_LIMITS, BASE_LIMITS
    

class Robot_Simulator:
    def __init__(self, l1,l2, HOME_THETA):
        self.l1 = l1
        self.l2 = l2
        self.theta = HOME_THETA
        self.duration = 0.0
        self.trajectory = []
        self.frame_index = 0
        self.N = 30
        self.target_height = 0.25
        self.fig = plt.figure()
        self.fig.subplots_adjust(bottom=0.25)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=25, azim=90)
        slider_ax = self.fig.add_axes([0.22, 0.14, 0.56, 0.03])
        self.elevation_slider = Slider(slider_ax, "View elevation", 0, 90, valinit=25, valfmt="%0.0f°")
        self.elevation_slider.on_changed(self.update_view)
        self.ax.mouse_init(rotate_btn=[], zoom_btn=[])
        self.ax.set_navigate(False)
        self.arm_line, = self.ax.plot([], [], [], "o-", markersize = 8, linewidth = 3)
        self.target_ring, = self.ax.plot([], [], [], color="tab:orange", linewidth=1.5, alpha=0.8)
        self.timer = self.fig.canvas.new_timer(interval=self.duration/self.N*1000)  # interval in milliseconds
        self.r = self.l1*jnp.cos(self.theta[1])+ self.l2*jnp.cos(self.theta[1] + self.theta[2])
        
    # Register its callback
        self.timer.add_callback(self.advance_frame)
        self.setup_temporary_controls()

    # Temporary test controls: remove this method and its call to remove them.
    def setup_temporary_controls(self):
        speed_ax = self.fig.add_axes([0.22, 0.08, 0.40, 0.03])
        self.speed_slider = Slider(speed_ax, "Speed", 0.25, 4.0, valinit=1.0, valfmt="%.2fx")
        self.speed_slider.on_changed(lambda value: self.set_animation_interval(value))

    def set_animation_interval(self, speed):
        self.timer.interval = max(1, round(self.duration / self.N * 1000 / speed))

    def update_view(self, elevation):
        self.ax.view_init(elev=elevation, azim=90)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        print("Click Detected")
        
        if event.inaxes != self.ax:
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        # Unproject two depths to form a world-space viewing ray, then
        # intersect it with the horizontal pickup/placement plane.
        inverse = np.linalg.inv(self.ax.get_proj())
        near = np.asarray(proj3d.inv_transform(event.xdata, event.ydata, -1, inverse)).reshape(3)
        far = np.asarray(proj3d.inv_transform(event.xdata, event.ydata, 1, inverse)).reshape(3)
        direction = far - near
        if abs(direction[2]) < 1e-10:
            print("Cannot select a point: viewing direction is parallel to the target plane")
            return
        point = near + (self.target_height - near[2]) / direction[2] * direction
        self.plan_move(*point)

    def plan_move(self, xtarget, ytarget, ztarget):
        angle = np.linspace(0, 2 * np.pi, 65)
        radius = 0.025 * (self.l1 + self.l2)
        self.target_ring.set_data_3d(xtarget + radius * np.cos(angle),
                                     ytarget + radius * np.sin(angle),
                                     np.full_like(angle, ztarget))
        self.fig.canvas.draw_idle()
        self.timer.stop()
        print(f"Target selected: ({xtarget:.3f}, {ytarget:.3f}, {ztarget:.3f})")

        try:
            self.trajectory, self.duration = optimizer(
                self.l1, self.l2, xtarget, ytarget, ztarget, self.N, np.asarray(self.theta)
            )
        except RuntimeError as exc:
            print(f"Could not plan trajectory: {exc}")
            return
    #= self.trajectory_generator(
    #self.theta, theta_target, self.num_steps)

        self.frame_index = 0
        self.set_animation_interval(getattr(self, "speed_slider", None).val if hasattr(self, "speed_slider") else 1.0)
        self.timer.start()
     

    def plot_arm(self):
        x, y, z = self.get_coordinates()
        self.arm_line.set_data_3d(x, y, z)



    def update_arm(self, theta_new):
        self.theta = jnp.array(theta_new)
        x, y, z = self.get_coordinates()
        self.arm_line.set_data_3d(x, y, z)
        self.fig.canvas.draw_idle()


    def get_coordinates(self):
        p2_coordinates = forward_kinematics(self.theta, self.l1, self.l2)
        x1 = self.l1 * jnp.cos(self.theta[1]) * jnp.cos(self.theta[0])
        y1 = self.l1 * jnp.cos(self.theta[1]) * jnp.sin(self.theta[0])
        z1 = self.l1 * jnp.sin(self.theta[1])
        p1_coordinates = jnp.array([x1, y1, z1])
        p0_coordinates = jnp.array([0, 0, 0])
        x_coordinates = jnp.array([p0_coordinates[0], p1_coordinates[0], p2_coordinates[0]])
        y_coordinates = jnp.array([p0_coordinates[1], p1_coordinates[1], p2_coordinates[1]])
        z_coordinates = jnp.array([p0_coordinates[2], p1_coordinates[2], p2_coordinates[2]])
        return x_coordinates, y_coordinates, z_coordinates



    def run(self):
        # Trace the four edges of the allowed joint-angle rectangle.
        samples = 300
        s0, s1 = SHOULDER_LIMITS
        e0, e1 = ELBOW_LIMITS
        shoulder = np.concatenate((np.linspace(s0, s1, samples),
                                   np.full(samples, s1),
                                   np.linspace(s1, s0, samples),
                                   np.full(samples, s0)))
        elbow = np.concatenate((np.full(samples, e0),
                                np.linspace(e0, e1, samples),
                                np.full(samples, e1),
                                np.linspace(e1, e0, samples)))
        
        radius = self.l1 * np.cos(shoulder) + self.l2 * np.cos(shoulder + elbow)
        z = self.l1 * np.sin(shoulder) + self.l2 * np.sin(shoulder + elbow)
        yaw = np.linspace(*BASE_LIMITS, 181)
        x = np.outer(np.cos(yaw), radius)
        y = np.outer(np.sin(yaw), radius)
        height = np.broadcast_to(z, x.shape).copy()
        height[(height < 0) | (y < -1e-10)] = np.nan
        self.ax.plot_surface(x, y, height, color="tab:blue", alpha=0.12, linewidth=0)
        reach = self.l1 + self.l2
        self.ax.set_xlim(-reach, reach)
        self.ax.set_ylim(0, reach)
        self.ax.set_zlim(0, reach)
        self.ax.set_box_aspect((2, 1, 1))
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z (height)")
        self.ax.grid(False)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)




        self.plot_arm()
        print("Arm plotted successfully")
        print("opening window")
        plt.show()
