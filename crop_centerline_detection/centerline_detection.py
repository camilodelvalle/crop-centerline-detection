
import sys
import cv2
from process_video import ProcessVideo


def main():
    
    pv = ProcessVideo(sys.argv[1], sys.argv[2]) # paths to input videos

    pv.new_video()
    
    while True: # main loop
        
        # read input video frames
        ret_rgb, frame_rgb = pv.video_rgb.read()
        ret_depth, frame_depth = pv.video_depth.read()
               
        if ret_rgb == True and ret_depth == True: 

            lat_dev, ang_dev, frame_rgb = pv.calculate_deviations(frame_depth, frame_rgb) 

            frame_rgb = pv.plot_labels(frame_rgb, lat_dev, ang_dev)

            pv.write_video(frame_rgb)

            # processed frames
            sys.stdout.write("\033[F\033[K\033[F\033[K")
            print(('\nProcessed frames: {0}/{1}').format(pv.processed_frames, pv.num_frames))
            
            # show resized image     
            frame_rgb = cv2.resize(frame_rgb, (640, 360), interpolation = cv2.INTER_AREA)
            cv2.imshow("Crop-row centerline detection", frame_rgb)
            key = cv2.waitKey(1)

            # end program
            if key == ord('q') or key == ord('Q'):
                pv.release_videos()

        else: 
            break
    
    pv.release_videos()


if __name__ == "__main__":
    main()