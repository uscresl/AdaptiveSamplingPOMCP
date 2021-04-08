import numpy as np

def link_3d_axes(fig,axes):
    def on_move(event):
        for ax in axes:
            if event.inaxes == ax:
                if ax.button_pressed in ax._rotate_btn:
                    for other_ax in axes:
                        if other_ax != ax:
                            other_ax.view_init(elev=ax.elev, azim=ax.azim)
                elif ax.button_pressed in ax._zoom_btn:
                    for other_ax in axes:
                        if other_ax != ax:
                            other_ax.set_xlim3d(ax.get_xlim3d())
                            other_ax.set_ylim3d(ax.get_ylim3d())
                            other_ax.set_zlim3d(ax.get_zlim3d())
        fig.canvas.draw_idle()
    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])