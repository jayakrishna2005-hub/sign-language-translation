import os
import cv2
cap=cv2.VideoCapture(0)
directory='Image/'
while True:
    _,frame=cap.read()
    count = {
             'A':len(os.listdir(directory+"/A")),
             'B':len(os.listdir(directory+"/B")),
             'C':len(os.listdir(directory+"/C")),
             'D':len(os.listdir(directory+"/D")),
             'E':len(os.listdir(directory+"/E")),
             'F':len(os.listdir(directory+"/F")),
             'G':len(os.listdir(directory+"/G")),
             'H':len(os.listdir(directory+"/H")),
             'I':len(os.listdir(directory+"/I")),
             'J':len(os.listdir(directory+"/J")),
             'K':len(os.listdir(directory+"/K")),
             'L':len(os.listdir(directory+"/L")),
             'M':len(os.listdir(directory+"/M")),
             'N':len(os.listdir(directory+"/N")),
             'O':len(os.listdir(directory+"/O")),
             'P':len(os.listdir(directory+"/P")),
             'Q':len(os.listdir(directory+"/Q")),
             'R':len(os.listdir(directory+"/R")),
             'S':len(os.listdir(directory+"/S")),
             'T':len(os.listdir(directory+"/T")),
             'U':len(os.listdir(directory+"/U")),
             'V':len(os.listdir(directory+"/V")),
             'W':len(os.listdir(directory+"/W")),
             'X':len(os.listdir(directory+"/X")),
             'Y':len(os.listdir(directory+"/Y")),
             'Z':len(os.listdir(directory+"/Z")),
             '1':len(os.listdir(directory+"/HELLO")),
             '2': len(os.listdir(directory + "/GOOD AFTERNOON")),
             '3': len(os.listdir(directory + "/HOW ARE YOU")),
             '4': len(os.listdir(directory + "/I AM FINE")),
             '5': len(os.listdir(directory + "/HAD YOUR LUNCH")),
             '6': len(os.listdir(directory + "/I ALSO HAD")),
             '7': len(os.listdir(directory + "/OKAY")),
             '8': len(os.listdir(directory + "/BYE")),
             '9': len(os.listdir(directory + "/,")),
             'a': len(os.listdir(directory + "/ALL THE BEST")),
             'b': len(os.listdir(directory + "/THANK YOU"))
             }
    
    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame,(0,40),(300,400),(255,255,255),2)
    cv2.imshow("data",frame)
    cv2.imshow("ROI",frame[40:400,0:300])
    frame=frame[40:400,0:300]
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('A'):
        cv2.imwrite(directory + 'A/' + str(count['A']) + '.png', frame)
    if interrupt & 0xFF == ord('B'):
        cv2.imwrite(directory + 'B/' + str(count['B']) + '.png', frame)
    if interrupt & 0xFF == ord('C'):
        cv2.imwrite(directory + 'C/' + str(count['C']) + '.png', frame)
    if interrupt & 0xFF == ord('D'):
        cv2.imwrite(directory + 'D/' + str(count['D']) + '.png', frame)
    if interrupt & 0xFF == ord('E'):
        cv2.imwrite(directory + 'E/' + str(count['E']) + '.png', frame)
    if interrupt & 0xFF == ord('F'):
        cv2.imwrite(directory + 'F/' + str(count['F']) + '.png', frame)
    if interrupt & 0xFF == ord('G'):
        cv2.imwrite(directory + 'G/' + str(count['G']) + '.png', frame)
    if interrupt & 0xFF == ord('H'):
        cv2.imwrite(directory + 'H/' + str(count['H']) + '.png', frame)
    if interrupt & 0xFF == ord('I'):
        cv2.imwrite(directory + 'I/' + str(count['I']) + '.png', frame)
    if interrupt & 0xFF == ord('J'):
        cv2.imwrite(directory + 'J/' + str(count['J']) + '.png', frame)
    if interrupt & 0xFF == ord('K'):
        cv2.imwrite(directory + 'K/' + str(count['K']) + '.png', frame)
    if interrupt & 0xFF == ord('L'):
        cv2.imwrite(directory + 'L/' + str(count['L']) + '.png', frame)
    if interrupt & 0xFF == ord('M'):
        cv2.imwrite(directory + 'M/' + str(count['M']) + '.png', frame)
    if interrupt & 0xFF == ord('N'):
        cv2.imwrite(directory + 'N/' + str(count['N']) + '.png', frame)
    if interrupt & 0xFF == ord('O'):
        cv2.imwrite(directory + 'O/' + str(count['O']) + '.png', frame)
    if interrupt & 0xFF == ord('P'):
        cv2.imwrite(directory + 'P/' + str(count['P']) + '.png', frame)
    if interrupt & 0xFF == ord('Q'):
        cv2.imwrite(directory + 'Q/' + str(count['Q']) + '.png', frame)
    if interrupt & 0xFF == ord('R'):
        cv2.imwrite(directory + 'R/' + str(count['R']) + '.png', frame)
    if interrupt & 0xFF == ord('S'):
        cv2.imwrite(directory + 'S/' + str(count['S']) + '.png', frame)
    if interrupt & 0xFF == ord('T'):
        cv2.imwrite(directory + 'T/' + str(count['T']) + '.png', frame)
    if interrupt & 0xFF == ord('U'):
        cv2.imwrite(directory + 'U/' + str(count['U']) + '.png', frame)
    if interrupt & 0xFF == ord('V'):
        cv2.imwrite(directory + 'V/' + str(count['V']) + '.png', frame)
    if interrupt & 0xFF == ord('W'):
        cv2.imwrite(directory + 'W/' + str(count['W']) + '.png', frame)
    if interrupt & 0xFF == ord('X'):
        cv2.imwrite(directory + 'X/' + str(count['X']) + '.png', frame)
    if interrupt & 0xFF == ord('Y'):
        cv2.imwrite(directory + 'Y/' + str(count['Y']) + '.png', frame)
    if interrupt & 0xFF == ord('Z'):
        cv2.imwrite(directory + 'Z/' + str(count['Z']) + '.png', frame)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'HELLO/'+str(count['1'])+'.png',frame)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory + 'GOOD AFTERNOON/' + str(count['2']) + '.png', frame)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory + 'HOW ARE YOU/' + str(count['3']) + '.png', frame)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory + 'I AM FINE/' + str(count['4']) + '.png', frame)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory + 'HAD YOUR LUNCH/' + str(count['5']) + '.png', frame)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory + 'I ALSO HAD/' + str(count['6']) + '.png', frame)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory + 'OKAY/' + str(count['7']) + '.png', frame)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory + 'BYE/' + str(count['8']) + '.png', frame)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory + 'SPACE/' + str(count['9']) + '.png', frame)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'ALL THE BEST/' + str(count['a']) + '.png', frame)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'THANK YOU/' + str(count['b']) + '.png', frame)


cap.release()
cv2.destroyAllWindows()