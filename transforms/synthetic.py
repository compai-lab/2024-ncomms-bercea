import numpy as np
import cv2
import torch

class SyntheticSprites:
    """
    Adds sprite abnormalities to the image
    """
    def __init__(self, intensity_ranges=[(1, 5), (30, 70), (95, 99)],
                 center_coord_range=(32, 96), axes_length_range=(4, 6), angle_range=(0, 360)): # 4,6
        """

        :param intensity_ranges: list of [tuples(mean, std)]
            mean and std for a normal distribution of intensities
        :param center_coord_range: tuple(min, max)
            min and max for an uniform distribution
        :param axes_length_range: tuple(mean, std)
            mean and std for a normal distribution of lesion size
        :param angle_range: tuple(min,max)
            min and max of the desired angle
        """
        super(SyntheticSprites, self).__init__()
        self.intensity_ranges = intensity_ranges
        self.center_coord_range = center_coord_range
        self.axes_length_range = axes_length_range
        self.angle_range = angle_range
        self.startAngle = 0
        self.endAngle = 360
        self.thickness = -1
        self.color_mask = 1

    def __call__(self, img):
        """
        :param img: 2D image
        :return: 2D image with abnormalities and 2D mask
        """
        # img = torch.tensor(img)
        new_axis = False
        if len(img.shape) > 2:
            new_axis = True
            img = img[0][0]
            img = img[:, :, np.newaxis]
        # device = img.get_device()
        # img = img.cpu().numpy()
        mask = np.zeros(img.shape)
        for i in range(len(self.intensity_ranges)):
            center_coordinatesAr = np.random.randint(self.center_coord_range[0], self.center_coord_range[1], 2)
            center_coordinates = (int(center_coordinatesAr[0]), int(center_coordinatesAr[1]))

            axesLengthAr = np.abs(np.random.normal(self.axes_length_range[0], self.axes_length_range[1], 2))
            axesLength = (int(max(axesLengthAr[0], 2)), int(max(axesLengthAr[1], 2))) # 2

            angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            # min_int, max_int = np.percentile(img, 98), 1
            # min_int, max_int = 0, np.percentile(img, 25)
            # min_int, max_int = np.percentile(img, 25), np.percentile(img, 75)
            min_int, max_int = np.percentile(img, self.intensity_ranges[i][0]), \
                               np.percentile(img, self.intensity_ranges[i][1])
            #
            color = np.minimum(1, np.abs(np.random.uniform(min_int, max_int)))
            # color = 1# np.random.randint(0*255, 1*255)/255
            # color = np.random.randint(0, 2)
            # color = np.minimum(1, np.abs(np.random.uniform(self.intensity_ranges[i][0], self.intensity_ranges[i][1])))
            #
            # color = 0
            img = cv2.ellipse(img, center_coordinates, axesLength, angle, self.startAngle, self.endAngle, color,
                              self.thickness)


            mask = cv2.ellipse(mask, center_coordinates, axesLength, angle, self.startAngle, self.endAngle,
                               self.color_mask, self.thickness)

        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        img = img[np.newaxis, :] if new_axis else img
        mask = mask[np.newaxis, :] if new_axis else mask

        # return torch.from_numpy(img).to(device), torch.from_numpy(mask).to(device)
        return img, mask

class GenerateMasks:
    # def __init__(self, min_size=20, max_size=35, center_coord_range=(32, 96), axes_length_range=(4, 8),
    #              angle_range=(0, 360)):
    def __init__(self, min_size=20, max_size=35, center_coord_range=(32, 96), axes_length_range=(2, 4),
                 angle_range=(0, 360)):
        super(GenerateMasks, self).__init__()
        self.size_range = (min_size, max_size)
        self.center_coord_range = center_coord_range
        self.axes_length_range = axes_length_range
        self.angle_range = angle_range
        self.startAngle = 0
        self.endAngle = 360
        self.thickness = -1
        self.color_mask = 1

    def __call__(self, x):
        """
        :param img: 2D image
        :return: 2D image with abnormalities and 2D mask
        """

        img_size = x.shape[-1]
        width = np.random.randint(self.size_range[0], self.size_range[1], (x.shape[0]))
        height = np.random.randint(self.size_range[0], self.size_range[1], (x.shape[0]))
        start_x = np.random.randint(int(img_size / 5), img_size - width - 1, (x.shape[0]))
        start_y = np.random.randint(int(img_size / 5), img_size - height - 1, (x.shape[0]))
        intensity = 1

        # # print(f'Synthetic Rectangle Generation: {width}- {height} + {intensity}')
        # #
        masks = []
        batches = x.shape[0]
        for b in range(batches):
            mask = np.expand_dims(np.zeros(x[b][0].shape), -1)

            for i in range(10): #5
                center_coordinatesAr = np.random.randint(self.center_coord_range[0], self.center_coord_range[1], 2)
                center_coordinates = (int(center_coordinatesAr[0]), int(center_coordinatesAr[1]))

                axesLengthAr = np.abs(np.random.normal(self.axes_length_range[0], self.axes_length_range[1], 2))
                axesLength = (int(max(axesLengthAr[0], 2)), int(max(axesLengthAr[1], 2)))

                angle = np.random.randint(self.angle_range[0], self.angle_range[1])

                mask = cv2.ellipse(mask, center_coordinates, axesLength, angle, self.startAngle, self.endAngle,
                                   intensity, self.thickness)
                # mask[start_x[b]:start_x[b] + width[b], start_y[b]:start_y[b] + height[b], 0] = intensity
            masks.append(np.transpose(mask, (2, 0, 1)))

        return torch.Tensor(np.asarray(masks)).to(x.get_device())


class SyntheticRect:
    def __init__(self, min_size=25, max_size=50):
        super(SyntheticRect, self).__init__()
        self.size_range = (min_size, max_size)
        self.synthSpr = SyntheticSprites(axes_length_range=(10, 12))

    def __call__(self, x):
        img_size = x.shape[-1]
        mask = np.zeros(x.shape)
        width = np.random.randint(self.size_range[0], self.size_range[1], (x.shape[0], x.shape[1]))
        height = np.random.randint(self.size_range[0], self.size_range[1], (x.shape[0], x.shape[1]))
        start_x = np.random.randint(int(img_size / 5), img_size - width - 1, (x.shape[0], x.shape[1]))
        start_y = np.random.randint(int(img_size / 5), img_size - height - 1, (x.shape[0], x.shape[1]))
        # intensity = np.random.uniform(0, 1, (x.shape[0], x.shape[1]))
        intensity = 0 #np.random.randint(int(np.percentile(x, 98) * 256) - 1 , 256, (x.shape[0], x.shape[1])) / 256.0


        # # print(f'Synthetic Rectangle Generation: {width}- {height} + {intensity}')
        # #
        for b in range(len(x)):
            x[b, 0, start_x[b, 0]:start_x[b, 0] + width[b, 0], start_y[b, 0]:start_y[b, 0] + height[b, 0]] = 0
            mask[b, 0,  start_x[b, 0]:start_x[b, 0] + width[b, 0], start_x[b, 0]:start_x[b, 0] + height[b, 0]] = 1

        # _, mask_spr = self.synthSpr(torch.zeros(x.shape).to(x.get_device()))
        # all_mask = mask_spr.cpu().numpy() + mask
        all_mask = mask
        all_mask[all_mask > 1] = 1

        return x

class CopyPaste:
    # def __init__(self, size_range=(25, 50)):
    def __init__(self, size_range=(25, 50)):
        super(CopyPaste, self).__init__()
        self.size_range = size_range
        self.ratio = 4

    def __call__(self, x):
        img_size = x.shape[-1]
        mask = np.zeros(x.shape)

        width = np.random.randint(self.size_range[0], self.size_range[1], (x.shape[0], x.shape[1]))
        height = np.random.randint(self.size_range[0], self.size_range[1], (x.shape[0], x.shape[1]))
        source_x = np.random.randint(int(img_size / self.ratio), img_size - width - 1, (x.shape[0], x.shape[1]))
        source_y = np.random.randint(int(img_size / self.ratio), img_size - height - 1, (x.shape[0], x.shape[1]))

        dest_x = np.random.randint(int(img_size / self.ratio), img_size - width - 1, (x.shape[0], x.shape[1]))
        dest_y = np.random.randint(int(img_size / self.ratio), img_size - height - 1, (x.shape[0], x.shape[1]))
        # # print(f'Synthetic Rectangle Generation: {width}- {height} + {intensity}')
        # #
        for b in range(len(x)):
            x[b, 0, dest_x[b, 0]:dest_x[b, 0] + width[b, 0], dest_y[b, 0]:dest_y[b, 0] + height[b, 0]] = \
                x[b, 0, source_x[b, 0]:source_x[b, 0] + width[b, 0], source_y[b, 0]:source_y[b, 0] + height[b, 0]]
            mask[b, 0,  dest_x[b, 0]:dest_x[b, 0] + width[b, 0], dest_y[b, 0]:dest_y[b, 0] + height[b, 0]] = 1
        return x, mask
