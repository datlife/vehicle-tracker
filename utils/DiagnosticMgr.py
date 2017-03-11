import cv2
import numpy as np
# Create better font
from PIL import ImageFont, ImageDraw, Image


class DiagnosticMgr(object):
    def __init__(self, img_filters, projection_mgr):
        self.filters = img_filters
        self.projection = projection_mgr
        self.font = "./docs/helvetica.ttf"

    def build(self, undst_img, lane_lines, bin_img,  edge_img, bird_eye_img, bird_eye_view, curved_fit, windows, curv, offset):

        # Assemble Diagnostic Screen
        diag_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        main_img = cv2.resize(lane_lines, (1600, 960), interpolation=cv2.INTER_AREA)
        # Main output image
        diag_screen = self._build_top_right(diag_screen, curved_fit, bird_eye_view, bird_eye_img, windows)
        diag_screen = self._build_bottom_right(diag_screen, undst_img, bin_img)
        diag_screen[0:960, 0:1600] = main_img
        diag_screen = self._build_status(diag_screen, main_img, curv, offset)
        return diag_screen

    def _build_top_right(self, diag_screen,  curved_fit, bird_eye_view, bird_eye_img, windows):

        histogram = self.cal_lane_prob(bird_eye_img)
        box_size = (320, 240)

        # Convert image to RGB
        # bird_eye_img = np.dstack((bird_eye_img, bird_eye_img, bird_eye_img)) * 255
        histogram = cv2.cvtColor(histogram, cv2.COLOR_GRAY2RGB)*255

        # Resize
        bird_eye_view = cv2.resize(bird_eye_view, (320, 240), interpolation=cv2.INTER_AREA)
        windows = cv2.resize(windows, (320, 240), interpolation=cv2.INTER_AREA)
        histogram = cv2.resize(histogram, (320, 240), interpolation=cv2.INTER_AREA) * 4
        curved_fit = cv2.resize(curved_fit, (320, 240), interpolation=cv2.INTER_AREA)*4

        # Add text
        bird_eye_view = self._build_title(bird_eye_view, "Bird Eye View", size=14, w_offset=2, h_offset=230)
        histogram = self._build_title(histogram, "Lane Line Probability", size=14, w_offset=2, h_offset=230)
        # Top right 1
        diag_screen[0:240, 1600:1920] = bird_eye_view
        diag_screen[240:480, 1600:1920] = windows
        diag_screen[480:720, 1600:1920] = histogram
        diag_screen[720:960, 1600:1920] = curved_fit

        # Top right 2
        return diag_screen

    def _build_bottom_right(self, diag_screen, lane_lines, bin_img):

        # Build debug images
        box_size = (320, 240)
        abs_sobel_grad = self.filters.abs_sobel_thresh(lane_lines, thresh_min=30, thresh_max=100)
        mag_sobel_grad = self.filters.mag_thresh(lane_lines, mag_thresh=(30, 100))
        s_thresh = self.filters.hls_select(lane_lines, thresh=(88, 250))
        h_thresh = self.filters.hls_select(lane_lines, thresh=(150, 255), channel=1)
        # Convert to RGB
        abs_sobel_grad = np.dstack((abs_sobel_grad, abs_sobel_grad, abs_sobel_grad)) * 255
        mag_sobel_grad = np.dstack((mag_sobel_grad, mag_sobel_grad, mag_sobel_grad)) * 255
        s_thresh = np.dstack((s_thresh, s_thresh, s_thresh)) * 255
        h_thresh = np.dstack((h_thresh, h_thresh, h_thresh)) * 255

        bin_img = np.dstack((bin_img, bin_img, bin_img))*255

        # Resize to fix small frame
        abs_sobel_grad = cv2.resize(abs_sobel_grad, box_size, interpolation=cv2.INTER_AREA)
        mag_sobel_grad = cv2.resize(mag_sobel_grad, box_size, interpolation=cv2.INTER_AREA)
        s_thresh = cv2.resize(s_thresh, box_size, interpolation=cv2.INTER_AREA)
        h_thresh = cv2.resize(h_thresh, box_size, interpolation=cv2.INTER_AREA)

        bin_img = cv2.resize(bin_img, box_size, interpolation=cv2.INTER_AREA)

        # Add Titles
        abs_sobel_grad = self._build_title(abs_sobel_grad, "Abs. Sobel Grad.", size=14, w_offset=120, h_offset=220)
        mag_sobel_grad = self._build_title(mag_sobel_grad, "Magnitude Sobel Grad.", size=14, w_offset=120, h_offset=220)
        s_thresh = self._build_title(s_thresh, "S Channel Thresh.", size=14, w_offset=120, h_offset=220)
        h_thresh = self._build_title(h_thresh, "H Channel Thresh.", size=14, w_offset=120, h_offset=220)
        bin_img = self._build_title(bin_img, "Combined Threshold", size=14, w_offset=120, h_offset=220)
        # Display
        diag_screen[840:1080, 0:320] = abs_sobel_grad
        diag_screen[840:1080, 320:640] = mag_sobel_grad
        diag_screen[840:1080, 640:960] = s_thresh
        diag_screen[840:1080, 960:1280] = h_thresh
        diag_screen[840:1080, 1280:1600] = bin_img

        # diag_screen[600:840, 1280:1600] = abs_sobel_grad
        # diag_screen[600:840, 1600:1920] = mag_sobel_grad
        # diag_screen[840:1080, 1280:1600] = s_thresh
        # diag_screen[840:1080, 1600:1920] = bin_img

        return diag_screen

    def _build_status(self, diag_screen, img, curv, offset):
        # Drawing text in diagnostic screen.
        status_screen = img

        if offset < 0.55:
            offset_str = str(round(offset, 3)) + " m"
        else:
            offset_str = str(round(offset, 3)) + " m " + "(Warning: Offset is too high!)"
        add1 = self._build_title(status_screen, "Estimated Center Offset: " + offset_str, size=30, w_offset=30, h_offset=30)
        status_screen = self._build_title(add1, "Estimated Radius of Curvature: " + str(round(curv, 3)) + " km", size=30, w_offset=30, h_offset=70)
        diag_screen[0:960, 0:1600] = status_screen  # Show curvature, offset
        return diag_screen

    def _build_title(self, img, text, size=20, w_offset=20, h_offset=20, color=(255, 255, 255, 255)):
        im = Image.fromarray(img, 'RGB')
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font=self.font, size=size)
        position = (w_offset, h_offset)
        draw.text(position, text, font=font, fill=color)
        img = np.array(im)
        return img

    def cal_lane_prob(self, bird_eye_img, size=(320, 240)):
        sample = cv2.resize(bird_eye_img, size, interpolation=cv2.INTER_AREA)
        histogram = np.zeros_like(sample)
        lane_probs = np.array(np.sum(sample[sample.shape[0] / 2:, :], axis=0), dtype='int32') + 50
        ploty = np.linspace(0, sample.shape[1] - 1, sample.shape[1])

        x = np.concatenate((ploty, ploty[::-1]), axis=0)
        y = np.concatenate((lane_probs - 3, lane_probs[::-1] + 3), axis=0)
        data = np.array(list(zip(x, y)), dtype='int32')
        cv2.fillPoly(histogram, [data], color=[255, 255, 0])
        histogram = cv2.flip(histogram, 0)
        histogram = np.uint8(255 * histogram)
        return histogram

