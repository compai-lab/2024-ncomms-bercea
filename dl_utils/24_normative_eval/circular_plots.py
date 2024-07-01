import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment


def add_labels(angles, values, labels, offset, ax):
    # This is the space between the end of the bar and the label
    padding = 4

    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle

        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle,
            y=value + padding,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

names = ['','AE-S', 'VAE', 'LTM', 'f-AnoGAN', 'SI-VAE', 'RA', 'DDPM-G', 'DDPM-S', 'ceVAE', 'MAE', 'MorphAEus',
         'pDDPM', 'PHANES', 'AutoDDPM', '', '']
values_rqi = [0, 0.88, 0.04, 0.17, 0.6, 0.64, 0.66, 0.75, 0.85, 0.14, 0.02, 0.56, 0.89, 0.89, 0.9, 0, 0]
# values_rqi = [0, 0.79, 0.39, 0.47, 0.36, 0.45, 0.77, 0.87, 0.78, 0.45, 0.43, 0.48, 0.77, 1.02, 1.16, 0, 0]
# values_ahi = [0, 0.02, 0.03, 0.07, 0.04, 0.06, 0.27, 0.15, 0.14, 0.09, 0.06, 0.03, 0.16, 0.43, 0.43, 0, 0]
values_ahi = [0,0,0,0,0,0,0.16,0.10,0.03,0,0,0,0.06,0.44,0.49,0,0]
values_caci = [0, 0.06, 0.22, 0.26, 0.15, 0.19, 0.22, 0.21, 0.31, 0.23, 0.20, 0.30, 0.28, 0.49, 0.45, 0,0]

            #[0, 1, 2, 3, 4, 5, 6, 7, 8,   9, 10,11,12, 13, 14..]
timestemps = [0, 1, 2, 9, 4, 5, 3, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16][::-1]

r1,g1,b1 = 255/255.0,127/255.0,80/255.0
r2,g2,b2 = 248/255.0,196/255.0,45/255.0
r3,g3,b3 = 72/255.0,209/255.0,204/255.0

colors_rqi = [(r1,g1,b1, min(values_rqi[ts],1)) for ts in timestemps]
colors_ahi = [(r2,g2,b2, min(values_ahi[ts]*2,1)) for ts in timestemps]
colors_caci = [(r3,g3,b3, min(values_caci[ts]*2,1)) for ts in timestemps]

# colors_rqi = [(r1,g1,b1, 0.6) for ts in timestemps]
# colors_ahi = [(r2,g2,b2, 0.6) for ts in timestemps]
# colors_caci = [(r3,g3,b3, 0.6) for ts in timestemps]

df_rqi = pd.DataFrame({
    "name": [names[ts] for ts in timestemps],
    "value": [values_rqi[ts] * 100 for ts in timestemps],
})
df_ahi = pd.DataFrame({
    "name": [names[ts] for ts in timestemps],
    "value": [values_ahi[ts] * 100 for ts in timestemps],
})
df_caci = pd.DataFrame({
    "name": [names[ts] for ts in timestemps],
    "value": [values_caci[ts] * 100 for ts in timestemps],
})

# df_rqi = pd.DataFrame({
#     "name": [names[ts] for ts in timestemps],
#     "value": [ 100 for ts in timestemps],
# })
# df_ahi = pd.DataFrame({
#     "name": [names[ts] for ts in timestemps],
#     "value": [ 100 for ts in timestemps],
# })
# df_caci = pd.DataFrame({
#     "name": [names[ts] for ts in timestemps],
#     "value": [100 for ts in timestemps],
# })


plt.close()
colors = ['#ff7f50', '#f8c42d', '#23b3b3']
offset_neg_rqi,offset_neg_ahi, offset_neg_caci = -100, -100, -100
offset_pos_rqi,offset_pos_ahi, offset_pos_caci=100,100,100

VALUES = df_rqi["value"].values
LABELS = df_rqi["name"].values
i = 0
df = pd.DataFrame({
    "name": [names[ts] for ts in timestemps],
    "value": [values[ts] * 100 for ts in timestemps],
})

ANGLES = np.linspace(0, 2 * np.pi, len(df), endpoint=False)

# Determine the width of each bar.
# The circumference is '2 * pi', so we divide that total width over the number of bars.
WIDTH = 2 * np.pi / len(VALUES)

# Determines where to place the first bar.
# By default, matplotlib starts at 0 (the first bar is horizontal)
# but here we say we want to start at pi/2 (90 deg)
OFFSET = np.pi / 2

# Initialize Figure and Axis

# fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20), subplot_kw={"projection": "polar"})
f = plt.figure(figsize=(10,10))
ax = f.add_axes([0.3, 0.3, 0.4, 0.4], polar=True) # Left, Bottom, Width, Height
ax2 = f.add_axes([0.125, 0.125, 0.75, 0.75], polar=True)
ax3 = f.add_axes([-0.025, -0.025, 1.05, 1.05], polar=True)

# Specify offset
ax.set_theta_offset(OFFSET)

# Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
ax.set_ylim(offset_neg_rqi,offset_pos_rqi)

# Remove all spines
ax.set_frame_on(False)

# Remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars
ax.bar(
    ANGLES, VALUES, width=WIDTH, linewidth=2,
    color=colors_rqi, edgecolor="white"
)

# Add labels
# add_labels(ANGLES, VALUES, LABELS, OFFSET, ax)
ax.set_theta_offset(OFFSET)


VALUES = df_ahi["value"].values
LABELS = df_ahi["name"].values
i = 1
df = pd.DataFrame({
    "name": [names[ts] for ts in timestemps],
    "value": [values[ts] * 100 for ts in timestemps],
})

# ax2.get_shared_x_axes().join(ax)
# Specify offset
ax2.set_theta_offset(OFFSET)

# Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
ax2.set_ylim(offset_neg_ahi,offset_pos_ahi)

# Remove all spines
ax2.set_frame_on(False)

# Remove grid and tick marks
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])

# Add bars
ax2.bar(
    ANGLES, VALUES, width=WIDTH, linewidth=2,
    color=colors_ahi, edgecolor="white"
)

# Add labels
# add_labels(ANGLES, VALUES, LABELS, OFFSET, ax2)
ax2.set_theta_offset(OFFSET)

VALUES = df_caci["value"].values
LABELS = df_caci["name"].values
i = 2
df = pd.DataFrame({
    "name": [names[ts] for ts in timestemps],
    "value": [values[ts] * 100 for ts in timestemps],
})

# ax2.get_shared_x_axes().join(ax)
# Specify offset
ax3.set_theta_offset(OFFSET)

# Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
ax3.set_ylim(offset_neg_caci,offset_pos_caci)

# Remove all spines
ax3.set_frame_on(False)

# Remove grid and tick marks
ax3.xaxis.grid(False)
ax3.yaxis.grid(False)
ax3.set_xticks([])
ax3.set_yticks([])

# Add bars
ax3.bar(
    ANGLES, VALUES, width=WIDTH, linewidth=2,
    color=colors_caci, edgecolor="white"
)

# Add labels
add_labels(ANGLES, VALUES, LABELS, OFFSET, ax3)
ax3.set_theta_offset(OFFSET)
plt.show()
# plt.savefig('./req_plot_updated.png', format='png', dpi=300, transparent=True)





#
#
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# def get_label_rotation(angle, offset):
#     # Rotation must be specified in degrees :(
#     rotation = np.rad2deg(angle + offset)
#     if angle <= np.pi:
#         alignment = "right"
#         rotation = rotation + 180
#     else:
#         alignment = "left"
#     return rotation, alignment
#
#
# def add_labels(angles, values, labels, offset, ax):
#     # This is the space between the end of the bar and the label
#     padding = 4
#
#     # Iterate over angles, values, and labels, to add all of them.
#     for angle, value, label, in zip(angles, values, labels):
#         angle = angle
#
#         # Obtain text rotation and alignment
#         rotation, alignment = get_label_rotation(angle, offset)
#
#         # And finally add the text
#         ax.text(
#             x=angle,
#             y=value + padding,
#             s=label,
#             ha=alignment,
#             va="center",
#             rotation=rotation,
#             rotation_mode="anchor"
#         )
#
# names = ['','AE-S', 'VAE', 'LTM', 'f-AnoGAN', 'SI-VAE', 'RA', 'DDPM-G', 'DDPM-S', 'ceVAE', 'MAE', 'MorphAEus',
#          'pDDPM', 'PHANES', 'AutoDDPM', '', '']
# values_rqi = np.sort([0, 0.88, 0.04, 0.17, 0.6, 0.64, 0.66, 0.75, 0.85, 0.14, 0.02, 0.56, 0.89, 0.89, 0.9, 0, 0])
# # values_rqi = [0, 0.79, 0.39, 0.47, 0.36, 0.45, 0.77, 0.87, 0.78, 0.45, 0.43, 0.48, 0.77, 1.02, 1.16, 0, 0]
# values_ahi = [0, 0.02, 0.03, 0.07, 0.04, 0.06, 0.27, 0.15, 0.14, 0.09, 0.06, 0.03, 0.16, 0.43, 0.43, 0, 0]
# values_caci = [0, 0.06, 0.22, 0.26, 0.15, 0.19, 0.22, 0.21, 0.31, 0.23, 0.20, 0.30, 0.28, 0.49, 0.45, 0,0]
#
#             #[0, 1, 2, 3, 4, 5, 6, 7, 8,   9, 10,11,12, 13, 14..]
# # timestemps = [0, 1, 2, 9, 4, 5, 3, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16][::-1]
# timestemps = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# r1,g1,b1 = 255/255.0,127/255.0,80/255.0
# r2,g2,b2 = 248/255.0,196/255.0,45/255.0
# r3,g3,b3 = 72/255.0,209/255.0,204/255.0
#
# colors_rqi = [(r1,g1,b1, min(values_rqi[ts],1)) for ts in timestemps]
# colors_ahi = [(r2,g2,b2, min(values_ahi[ts]*2,1)) for ts in timestemps]
# colors_caci = [(r3,g3,b3, min(values_caci[ts]*2,1)) for ts in timestemps]
#
# # colors_rqi = [(r1,g1,b1, 0.6) for ts in timestemps]
# # colors_ahi = [(r2,g2,b2, 0.6) for ts in timestemps]
# # colors_caci = [(r3,g3,b3, 0.6) for ts in timestemps]
#
# df_rqi = pd.DataFrame({
#     "name": [names[ts] for ts in timestemps],
#     "value": [values_rqi[ts] * 100 for ts in timestemps],
# })
# df_ahi = pd.DataFrame({
#     "name": [names[ts] for ts in timestemps],
#     "value": [values_ahi[ts] * 100 for ts in timestemps],
# })
# df_caci = pd.DataFrame({
#     "name": [names[ts] for ts in timestemps],
#     "value": [values_caci[ts] * 100 for ts in timestemps],
# })
#
# # df_rqi = pd.DataFrame({
# #     "name": [names[ts] for ts in timestemps],
# #     "value": [ 100 for ts in timestemps],
# # })
# # df_ahi = pd.DataFrame({
# #     "name": [names[ts] for ts in timestemps],
# #     "value": [ 100 for ts in timestemps],
# # })
# # df_caci = pd.DataFrame({
# #     "name": [names[ts] for ts in timestemps],
# #     "value": [100 for ts in timestemps],
# # })
#
#
# plt.close()
# colors = ['#ff7f50', '#f8c42d', '#23b3b3']
# offset_neg_rqi,offset_neg_ahi, offset_neg_caci = -100, -100, -100
# offset_pos_rqi,offset_pos_ahi, offset_pos_caci=100,100,100
#
#
# plt.close()
# VALUES = df_rqi["value"].values
# LABELS = df_rqi["name"].values
# i = 0
# df = pd.DataFrame({
#     "name": [names[ts] for ts in timestemps],
#     "value": [values[ts] * 100 for ts in timestemps],
# })
#
# ANGLES = np.linspace(0, 2 * np.pi, len(df), endpoint=False)
#
# # Determine the width of each bar.
# # The circumference is '2 * pi', so we divide that total width over the number of bars.
# WIDTH = 2 * np.pi / len(VALUES)
#
# # Determines where to place the first bar.
# # By default, matplotlib starts at 0 (the first bar is horizontal)
# # but here we say we want to start at pi/2 (90 deg)
# OFFSET = np.pi / 2
#
# # Initialize Figure and Axis
#
# # fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20), subplot_kw={"projection": "polar"})
# f = plt.figure(figsize=(10,10))
# ax = f.add_axes([0.3, 0.3, 0.4, 0.4], polar=False) # Left, Bottom, Width, Height
# # ax2 = f.add_axes([0.125, 0.125, 0.75, 0.75], polar=True)
# # ax3 = f.add_axes([-0.025, -0.025, 1.05, 1.05], polar=True)
#
# # Specify offset
# # ax.set_theta_offset(OFFSET)
#
# # Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
# ax.set_ylim(offset_neg_rqi,offset_pos_rqi)
#
# # Remove all spines
# ax.set_frame_on(False)
#
# # Remove grid and tick marks
# ax.xaxis.grid(False)
# ax.yaxis.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
#
# # Add bars
# ax.bar(
#     ANGLES, VALUES, width=WIDTH, linewidth=2,
#     color=colors_rqi, edgecolor="white"
# )
#
# # Add labels
# # add_labels(ANGLES, VALUES, LABELS, OFFSET, ax)
# # ax.set_theta_offset(OFFSET)
#
#
# # VALUES = df_ahi["value"].values
# # LABELS = df_ahi["name"].values
# # i = 1
# # df = pd.DataFrame({
# #     "name": [names[ts] for ts in timestemps],
# #     "value": [values[ts] * 100 for ts in timestemps],
# # })
# #
# # # ax2.get_shared_x_axes().join(ax)
# # # Specify offset
# # ax2.set_theta_offset(OFFSET)
# #
# # # Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
# # ax2.set_ylim(offset_neg_ahi,offset_pos_ahi)
# #
# # # Remove all spines
# # ax2.set_frame_on(False)
# #
# # # Remove grid and tick marks
# # ax2.xaxis.grid(False)
# # ax2.yaxis.grid(False)
# # ax2.set_xticks([])
# # ax2.set_yticks([])
# #
# # # Add bars
# # ax2.bar(
# #     ANGLES, VALUES, width=WIDTH, linewidth=2,
# #     color=colors_ahi, edgecolor="white"
# # )
# #
# # # Add labels
# # # add_labels(ANGLES, VALUES, LABELS, OFFSET, ax2)
# # ax2.set_theta_offset(OFFSET)
# #
# # VALUES = df_caci["value"].values
# # LABELS = df_caci["name"].values
# # i = 2
# # df = pd.DataFrame({
# #     "name": [names[ts] for ts in timestemps],
# #     "value": [values[ts] * 100 for ts in timestemps],
# # })
# #
# # # ax2.get_shared_x_axes().join(ax)
# # # Specify offset
# # ax3.set_theta_offset(OFFSET)
# #
# # # Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
# # ax3.set_ylim(offset_neg_caci,offset_pos_caci)
# #
# # # Remove all spines
# # ax3.set_frame_on(False)
# #
# # # Remove grid and tick marks
# # ax3.xaxis.grid(False)
# # ax3.yaxis.grid(False)
# # ax3.set_xticks([])
# # ax3.set_yticks([])
# #
# # # Add bars
# # ax3.bar(
# #     ANGLES, VALUES, width=WIDTH, linewidth=2,
# #     color=colors_caci, edgecolor="white"
# # )
# #
# # # Add labels
# # # add_labels(ANGLES, VALUES, LABELS, OFFSET, ax3)
# # ax3.set_theta_offset(OFFSET)
# # plt.show()
# plt.savefig('./req_plot_rqi.png', format='png', dpi=300, transparent=True)