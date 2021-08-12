
import numpy as np
import cv2
import sys
import os


class ProcessVideo:
    """
    Class to process video files (rgb and depth) and detect the crop-row centerline
    """

    def __init__(self, path_video_color, path_video_depth):

        self.video_rgb = cv2.VideoCapture(path_video_color)
        self.video_depth = cv2.VideoCapture(path_video_depth)

        self.processed_frames = 0
        self.num_frames = int(self.video_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video_rgb.get(cv2.CAP_PROP_FPS))

        self.height, self.width = (int(self.video_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                   int(self.video_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)))


    def new_video(self):
        """
        Write a new output video
        """
        
        if os.path.exists('../output_videos') == False:
            os.mkdir('../output_videos')
        try:
            num_file = np.load('../output_videos/num_file.npy')
            num_file += 1
            np.save('../output_videos/num_file.npy', num_file)
        except:
            np.save('../output_videos/num_file.npy', 0)
            num_file = 0

        self.video_output = cv2.VideoWriter('../output_videos/output_video_' + str(num_file) + '.mp4',
                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.fps, (self.width, self.height))


    def write_video(self, frame):
        """
        Write frame to output video
        """
        
        self.processed_frames += 1
        self.video_output.write(frame)


    def calculate_deviations(self, frame_depth, frame_rgb):
        """
        Calculate deviations using computer vision techniques
        """

        frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_RGB2GRAY)
        
        # binary segmentation (Otsu binarization)
        otsu_regions = [] 
        for i in range(0,self.height):
            _, otsu_region = cv2.threshold(frame_depth[i:1+i,:],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            otsu_regions.append(otsu_region)
        img_otsu = np.concatenate(otsu_regions, axis=0)

        # Laplacian filter
        img_laplacian = cv2.Laplacian(img_otsu,cv2.CV_8U,6) 
        
        img_edges = np.zeros((self.height,self.width,1), np.uint8) 
        xp, yp = [], []

        line_width = 5 

        # edge detection and noise removal
        for i in range(0,self.height): 
            
            points_left = np.asarray(np.where(img_laplacian[i][0:int(self.width/2)]==255))
            points_right = np.asarray(np.where(img_laplacian[i][int(self.width/2):self.width-1]==255))

            if len(points_left[0])!=0 and len(points_right[0])!=0:
                
                pt_left = points_left[0][-1]
                pt_right = points_right[0][0]+int(self.width/2)

                img_edges[i][pt_left][0]=255 
                img_edges[i][pt_right][0]=255

                frame_rgb[i][pt_left:pt_left+line_width][0:]=255 
                frame_rgb[i][pt_right:pt_right+line_width][0:]=255 

                m = int((pt_left+pt_right)/2)

                xp.append(i)
                yp.append(m) 

        # linear regression
        a = np.vstack([xp,np.ones(len(xp))]).T
        m, b = np.linalg.lstsq(a,yp,rcond=None)[0]

        center_width = int(self.width/2) 
        center_centerline = m*(self.height/2)+b

        lat_dev = center_centerline-center_width # lateral deviation [pixels]
        ang_dev = -np.arctan(m) # angular deviation [rad]

        return lat_dev, ang_dev, frame_rgb


    def plot_labels(self, frame_rgb, lat_dev, ang_dev):
        """
        Plot lines and labels
        """

        # plot central axes
        cv2.line(frame_rgb,(int(self.width/2),int(self.height/2)-50),(int(self.width/2),int(self.height/2)+50),(255, 255, 255),4)
        cv2.line(frame_rgb,(int(self.width/2)-50,int(self.height/2)),(int(self.width/2)+50,int(self.height/2)),(255, 255, 255),4)

        # plot crop-row centerline
        x1=(self.width/2) + lat_dev + (self.height/2)*np.tan(ang_dev)
        x2=(self.width/2) + lat_dev - (self.height/2)*np.tan(ang_dev)
        cv2.line(frame_rgb,(int(x1),int(0)),(int(x2),int(self.height)),(50,255,50),18)
        cv2.line(frame_rgb,(int(self.width/2),int(self.height/2)), (int(lat_dev+(self.width/2)),int(self.height/2)),(0,0,255),8)
        cv2.line(frame_rgb,(int(self.width/2),int(self.height/2-30)),(int(self.width/2),int(self.height/2+30)),(0,0,255),8)
        cv2.line(frame_rgb,(int(lat_dev+(self.width/2)),int(self.height/2-30)),(int(lat_dev+(self.width/2)),int(self.height/2+30)),(0,0,255),8)            

        # plot bottom bar
        cv2.rectangle(frame_rgb, (0,self.height-60), (self.width, self.height), (255, 255, 255), -1)
        cv2.putText(frame_rgb,'Lateral [px]: '+str(int(-lat_dev)),(200,self.height-20),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0), 2)
        cv2.putText(frame_rgb,'Angular [deg]: '+str(int(-ang_dev*(180/np.pi))),(700,self.height-20),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0), 2)
        
        return frame_rgb


    def release_videos(self):
        """
        Release video files
        """

        cv2.destroyAllWindows()

        self.video_rgb.release()
        self.video_depth.release()
        self.video_output.release()

        print('\n***** FINISHED *****\n')
        exit(1)
