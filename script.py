import cv2

# define a video capture object
n_classes = int(input('numero de clases clases:'))
n_instances = int(input('ingrese el numero de instancias por clase:'))

for i in range(n_classes):
    nombre_clase = input("ingrese el nombre de la clase:")
    vid = cv2.VideoCapture(0)
    j=0
    nums = 0

    while(True):
            
        # Capture the video frame by frame 
        ret, frame = vid.read() 
        
        # Display the resulting frame 
        cv2.imshow('frame', frame)
        if j%30 == 0:
            nums+=1
            print(nums)
            cv2.imwrite(str(nombre_clase)+'_'+str(nums)+'.jpg', frame)
        j+=1
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        if nums >= n_instances:
            break
        
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 

